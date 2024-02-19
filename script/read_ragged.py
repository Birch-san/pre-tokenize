import argparse
from pathlib import Path
from typing import List, Callable, NamedTuple, Generator, Iterable
import fnmatch
import re
from os import listdir
from numpy.typing import NDArray
import numpy as np
from transformers import T5TokenizerFast

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

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--in-dir', type=Path, help='directory in which *.{data,len}.npy reside. download from https://huggingface.co/datasets/Birchlabs/c4-t5-ragged/tree/main/en/validation')
  args = p.parse_args()
  shards: List[str] = locate_shards(args.in_dir)

  tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('google/t5-v1_1-base', legacy=False)

  for data_path in shards:
    print(f'=Shard {data_path}=')
    rarr: RaggedArray = read_file(data_path)
    samples: Iterable[NDArray] = read_shard_contents(rarr)
    for samp_ix, sample in enumerate(samples):
      text: str = tokenizer.decode(sample)
      print(f'==Sample {samp_ix}==\n{text}')
      pass # put your breakpoint here if you're debugging