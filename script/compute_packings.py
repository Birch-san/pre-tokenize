import argparse
from pathlib import Path
from typing import List, Callable, NamedTuple, Generator, Iterable
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

def wrap_lengths(len_in: NDArray, context_len: int, length_slab_size: int, add_bos=False) -> NDArray:
  bytes_per_int16 = np.dtype(np.int16).itemsize
  len_out = np.zeros((length_slab_size // bytes_per_int16,), dtype=np.int16)
  len_ptr = 0

  sample_max_len = context_len-1 if add_bos else context_len
  for len_ in len_in:
    # amount of sample remaining
    sample_len: int | NDArray = len_
    while sample_len > sample_max_len:
      len_out[len_ptr] = context_len
      len_ptr += 1
      # probably cheaper to reassign python ints than numpy array ints
      sample_len = int(sample_len) - sample_max_len
    len_out[len_ptr] = sample_len+1 if add_bos else sample_len
    len_ptr += 1

  len_out = len_out[:len_ptr]
  return len_out

def compute_packings(len_np: NDArray, context_len: int, device=torch.device('cpu')) -> Packing:
  len_t: LongTensor = torch.as_tensor(len_np, device=device, dtype=torch.long)
  # we are not expecting any sample to be as short as 0
  assert torch.all(len_t >= 1).item()

  histogram: LongTensor = torch.histc(len_t, bins=context_len, min=0, max=context_len)
  packings: Packing = pack_using_nnlshp(histogram.cpu().numpy(), max_sequence_length=context_len, max_sequences_per_pack=2)
  return packings

def pack_shard(len_path: Path, context_len: int, length_slab_size: int, add_bos=False, device=torch.device('cpu')):
  lengths: NDArray = np.load(str(len_path), allow_pickle=False)
  wrapped_lengths: NDArray = wrap_lengths(lengths, context_len, length_slab_size, add_bos=add_bos)
  strategy_set, strategy_repeat_count = compute_packings(wrapped_lengths, context_len, device=device)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--add-bos', action='store_true', help="whether to increase each sample's length by 1 to account for a BOS's being added.")
  p.add_argument('--context-len', type=int, help="sequence length into which samples will be packed")
  p.add_argument('--in-dir', type=Path, help='directory in which *.{data,len}.npy reside. download from https://huggingface.co/datasets/Birchlabs/c4-t5-ragged/tree/main/en/validation')
  p.add_argument('--length-slab-size', default=4*1024**2, type=int, help='bytes to allocate for computing wrapped lengths')
  args = p.parse_args()
  shards: List[str] = locate_shards(args.in_dir)

  device=torch.device('cuda')

  for data_path in shards:
    len_path: Path = data_path.with_suffix('').with_suffix('.len.npy')
    print(f'=Shard {len_path}=')
    pack_shard(len_path, args.context_len, args.length_slab_size, add_bos=args.add_bos, device=device)
    
    