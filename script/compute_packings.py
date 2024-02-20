import argparse
from pathlib import Path
from typing import List, Callable, NamedTuple, Generator, Set, Iterable
import fnmatch
import re
from os import listdir
from numpy.typing import NDArray
import numpy as np
import torch
from torch import LongTensor
from transformers import T5TokenizerFast

from pre_tokenize.packed_bert.nnlshp import pack_using_nnlshp, Packing

class RaggedArray(NamedTuple):
  data: NDArray
  lens: NDArray
  indices: NDArray

class DataAndIx(NamedTuple):
  data: NDArray
  indices: NDArray

class PackingAndHisto(NamedTuple):
  packing: Packing
  histogram: LongTensor

def read_sample(rarr: DataAndIx, ix: int) -> NDArray:
  data, indices = rarr
  start_ix, end_ix = indices[ix], indices[ix+1]
  slice: NDArray = data[start_ix:end_ix]
  return slice

def read_file(data_path: Path) -> RaggedArray:
  len_path: Path = data_path.with_suffix('').with_suffix('.len.npy')
  data: NDArray = np.load(str(data_path), mmap_mode='r', allow_pickle=False)
  # length of each sample in the ragged array.
  # in our situation (reading samples from the array) this isn't really what we want.
  lens: NDArray = np.load(str(len_path), allow_pickle=False)

  # compute indices from lengths via cumsum(). but prepend a 0-index.
  # int32 indexing assumes data shard size ≤ 2GiB
  indices: NDArray = np.zeros((lens.shape[0]+1,), dtype=np.int32)
  indices[1:] = lens.cumsum()
  return RaggedArray(data, lens, indices)

def locate_shards(dir: Path) -> List[str]:
  shards_unsorted: List[str] = fnmatch.filter(listdir(str(dir)), f'*.data.npy')
  matcher = f'.(\d+)-of-(\d+).data.npy$'
  get_shard_ix: Callable[[str], int] = lambda fname: int(re.search(matcher, fname).group(1))
  keyer: Callable[[str], int] = lambda fname: get_shard_ix(Path(fname).name)
  shards: List[str] = sorted(shards_unsorted, key=keyer)
  return [dir / shard for shard in shards]

def read_shard_contents(rarr: RaggedArray) -> Generator[NDArray, None, None]:
  data, _, indices = rarr
  data_and_ix = DataAndIx(data, indices)
  # we could equally read sample_count from lens.shape[0], but I prefer to shorten the lifetime of arrays I'm not using
  sample_count: int = indices.shape[0]-1
  for samp_ix in range(sample_count):
    tokens: NDArray = read_sample(data_and_ix, samp_ix)
    yield tokens

def compute_packings(len_np: NDArray, context_len: int, device=torch.device('cpu')) -> PackingAndHisto:
  len_t: LongTensor = torch.as_tensor(len_np, device=device, dtype=torch.long)
  # we are not expecting any sample to be as short as 0
  assert torch.all(len_t >= 1).item()

  histogram: LongTensor = torch.histc(len_t, bins=context_len, min=0, max=context_len)
  packings: Packing = pack_using_nnlshp(histogram.cpu().numpy(), max_sequence_length=context_len, max_sequences_per_pack=2)
  return PackingAndHisto(packings, histogram)

def pack_shard(rarr: RaggedArray, context_len: int, data_slab_size: int, length_slab_size: int, device=torch.device('cpu')):
  in_data, lengths, indices = rarr
  packings, _ = compute_packings(lengths, context_len, device=device)
  strategy_set, strategy_repeat_count = packings
  ixs_remaining: Set[int] = set()
  # we prepend an unused 0-len set just to ensure we can 0-index into it
  samples_by_len: List[Set[int]] = [set(), *(set() for _ in range(context_len))]
  for ix, len_ in enumerate(lengths):
    samples_by_len[len_].add(ix)
    ixs_remaining.add(ix)

  bytes_per_int16 = np.dtype(np.int16).itemsize
  out_data = np.zeros((data_slab_size // bytes_per_int16,), dtype=np.int16)
  out_data_ptr = 0
  out_len_subsample = np.zeros((length_slab_size // bytes_per_int16,), dtype=np.int32)
  out_len_subsample_ptr = 0
  out_len_sample = np.zeros((length_slab_size // bytes_per_int16,), dtype=np.int32)
  out_len_sample_ptr = 0

  in_data_ptr = 0
  for ix, len_ in enumerate(lengths):
    cumlen = 0

    strategy_ix: int = (len_ if len_ < context_len//2 else context_len-len_)-1
    strategy: List[int] = strategy_set[strategy_ix]
    strategy_count: int = strategy_repeat_count[strategy_ix].item()
    assert strategy_count > 0
    strategy_repeat_count[strategy_ix] -= 1

    if ix in ixs_remaining:
      out_data[out_data_ptr:out_data_ptr+len_] = in_data[in_data_ptr:in_data_ptr+len_]
      ixs_remaining.remove(ix)
      samples_by_len[len_].remove(ix)
      out_data_ptr += len_
      out_len_subsample[out_len_subsample_ptr] = len_
      out_len_subsample_ptr += 1
      cumlen += len_

    # we assume max_sequences_per_pack=2
    partner_len: int = strategy[1 if len_ < context_len//2 else 0]
    samples_by_partner_len: Set[int] = samples_by_len[partner_len]
    if samples_by_partner_len:
      partner_ix: int = next(iter(samples_by_partner_len))
      partner_ptr: int = indices[partner_ix].item()
      partner_len_: int = lengths[partner_ix].item()
      assert partner_len_ == partner_len_
      out_data[out_data_ptr:out_data_ptr+partner_len_] = in_data[partner_ptr:partner_ptr+partner_len_]
      ixs_remaining.remove(partner_ix)
      samples_by_partner_len.remove(partner_ix)
      out_data_ptr += partner_len_
      out_len_subsample[out_len_subsample_ptr] = partner_len_
      out_len_subsample_ptr += 1
      cumlen += partner_len_
    
    if cumlen > 0:
      out_len_sample[out_len_sample_ptr] = cumlen
      out_len_sample_ptr += 1

    in_data_ptr += len_
  pass
  



if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--context-len', type=int, help="sequence length into which samples will be packed")
  p.add_argument('--in-dir', type=Path, help='directory in which *.{data,len}.npy reside. download from https://huggingface.co/datasets/Birchlabs/c4-t5-ragged/tree/main/en/validation')
  p.add_argument('--data-slab-size', default=768*1024**2, type=int, help='bytes to allocate for ragged array data')
  p.add_argument('--length-slab-size', default=4*1024**2, type=int, help='bytes to allocate for ragged array indices')
  args = p.parse_args()
  shards: List[str] = locate_shards(args.in_dir)

  device=torch.device('cuda')

  for data_path in shards:
    # len_path: Path = data_path.with_suffix('').with_suffix('.len.npy')
    print(f'=Shard {data_path}=')
    rarr: RaggedArray = read_file(data_path)
    pack_shard(rarr, args.context_len, args.data_slab_size, args.length_slab_size, device=device)
    
    