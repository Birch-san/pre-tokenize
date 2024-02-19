import argparse
from pathlib import Path
from typing import List, Callable, NamedTuple, Generator
import fnmatch
import re
from os import makedirs, listdir
from numpy.typing import NDArray
import numpy as np

class DataAndLen(NamedTuple):
  data: NDArray
  lens: NDArray

def read_file(data_path: Path) -> DataAndLen:
  len_path: Path = data_path.with_suffix('').with_suffix('.len.npy')
  data: NDArray = np.load(str(data_path), mmap_mode='r', allow_pickle=False)
  # length of each sample in the ragged array.
  lens: NDArray = np.load(str(len_path), allow_pickle=False)
  return DataAndLen(data, lens)

def locate_shards(dir: Path) -> List[str]:
  shards_unsorted: List[str] = fnmatch.filter(listdir(str(dir)), f'*.data.npy')
  matcher = f'.(\d+)-of-(\d+).data.npy$'
  get_shard_ix: Callable[[str], int] = lambda fname: int(re.search(matcher, fname).group(1))
  keyer: Callable[[str], int] = lambda fname: get_shard_ix(Path(fname).name)
  shards: List[str] = sorted(shards_unsorted, key=keyer)
  return [dir / shard for shard in shards]

def do_wrap(
  data_and_len: DataAndLen,
  out_data_path: Path,
  context_len: int,
  data_slab_size: int,
  length_slab_size: int,
  add_bos=False,
) -> None:
  bytes_per_int16 = np.dtype(np.int16).itemsize
  data_out = np.zeros((data_slab_size // bytes_per_int16,), dtype=np.int16)
  data_ptr = data_in_ptr = 0
  len_out = np.zeros((length_slab_size // bytes_per_int16,), dtype=np.int16)
  len_ptr = 0

  data_in, len_in = data_and_len
  
  sample_max_len = context_len-1 if add_bos else context_len
  for len_ in len_in:
    # amount of sample remaining
    sample_len: int | NDArray = len_
    while sample_len > sample_max_len:
      len_out[len_ptr] = context_len
      len_ptr += 1
      # if add_bos:
      #   # if we were re-using the buffer, we would want to make this explicit.
      #   # but since we use a new data buffer each time, we rely on the fact that it's been zeroed-out
      #   data_out[data_ptr] = 0
      samp_start_ptr = data_ptr+1 if add_bos else data_ptr
      data_out[samp_start_ptr:data_ptr+context_len] = data_in[data_in_ptr:data_in_ptr+sample_max_len]
      data_in_ptr += sample_max_len
      data_ptr += context_len
      # probably cheaper to reassign python ints than numpy array ints
      sample_len = int(sample_len) - sample_max_len
    len_out[len_ptr] = sample_len+1 if add_bos else sample_len
    len_ptr += 1
    if add_bos:
      # if we were re-using the buffer, we would want to make this explicit.
      # but since we use a new data buffer each time, we rely on the fact that it's been zeroed-out
      # data_out[data_ptr] = 0
      data_out[data_ptr+1:data_ptr+1+sample_len] = data_in[data_in_ptr:data_in_ptr+sample_len]
      data_ptr += sample_len + 1
    else:
      data_out[data_ptr:data_ptr+sample_len] = data_in[data_in_ptr:data_in_ptr+sample_len]
      data_ptr += sample_len
    data_in_ptr += sample_len

  len_out = len_out[:len_ptr]
  data_out = data_out[:data_ptr]

  np.save(str(out_data_path), data_out, allow_pickle=False)
  print(f'Saved {str(out_data_path)}', flush=True)

  len_path: Path = out_data_path.with_suffix('').with_suffix('.len.npy')
  np.save(str(len_path), len_out, allow_pickle=False)
  print(f'Saved {str(len_path)}', flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--add-bos', action='store_true', help="whether to increase each sample's length by 1 to account for a BOS's being added.")
  p.add_argument('--context-len', type=int, help="sequence length into which samples will be packed")
  p.add_argument('--in-dir', type=Path, help='directory in which *.{data,len}.npy reside. download from https://huggingface.co/datasets/Birchlabs/c4-t5-ragged/tree/main/en/validation')
  p.add_argument('--out-dir', default=Path('out-wrap'), type=Path, help='directory into which to output wrapped ragged arrays')
  p.add_argument('--data-slab-size', default=512*1024**2, type=int, help='bytes to allocate for ragged array data')
  p.add_argument('--length-slab-size', default=4*1024**2, type=int, help='bytes to allocate for ragged array lengths')
  args = p.parse_args()
  shards: List[str] = locate_shards(args.in_dir)

  makedirs(str(args.out_dir), exist_ok=True)

  def wrap_shard(data_path: Path) -> None:
    data_and_len: DataAndLen = read_file(data_path)
    out_shard_path: Path = args.out_dir / data_path.name
    if out_shard_path.is_file():
      print(f'Skipping because {str(out_shard_path)} already exists.')
      return
    do_wrap(
      data_and_len,
      out_shard_path,
      args.context_len,
      args.data_slab_size,
      args.length_slab_size,
      add_bos=args.add_bos,
    )

  for data_path in shards:
    print(f'=Shard {data_path}=')
    wrap_shard(data_path)