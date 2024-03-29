from pathlib import Path
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool, AsyncResult
import fnmatch
from os import makedirs, listdir
import argparse
from typing import List, Callable, Set, Dict, Any, Optional, TYPE_CHECKING
import re
from threading import Thread
import gzip
import json

def int_to_file(int_samps: Queue, out_shard_path: Path, data_slab_size: int, length_slab_size: int) -> None:
  import numpy as np
  from numpy.typing import NDArray
  print('int_to_file: Running', flush=True)

  bytes_per_int16 = np.dtype(np.int16).itemsize
  bytes_per_int32 = np.dtype(np.int32).itemsize
  arr = np.zeros((data_slab_size // bytes_per_int16,), dtype=np.int16)
  ptr = 0
  # yes there appear to be documents larger than an int16's quantity of tokens
  lengths = np.zeros((length_slab_size // bytes_per_int32,), dtype=np.int32)
  len_ptr = 0

  def do_task() -> bool:
    nonlocal ptr, len_ptr
    samp: Optional[NDArray] = int_samps.get()
    if samp is None:
      return False
    samp_len: int = samp.shape[-1]
    if ptr + samp_len > arr.shape[-1]:
      raise OverflowError(f'Cannot write sample (len {samp_len} * item size {arr.itemsize} = {samp_len * arr.itemsize} bytes) into {arr.nbytes / 1024**2:2f}MiB slab, with {(arr.shape[-1]-ptr)*arr.itemsize} bytes remaining. You should increase data slab size.')
    if len_ptr + 1 > lengths.shape[-1]:
      raise OverflowError(f'Cannot write sample length ({lengths.itemsize} bytes) into {lengths.nbytes / 1024**2:2f}MiB slab, with {(lengths.shape[-1]-len_ptr)*lengths.itemsize} bytes remaining. You should increase index slab size.')
    arr[ptr:ptr+samp_len] = samp
    lengths[len_ptr] = samp_len
    ptr += samp_len
    len_ptr += 1
    return True

  while do_task(): pass

  arr = arr[:ptr]
  lengths = lengths[:len_ptr]

  data_path: Path = out_shard_path.with_suffix('.data.npy')
  np.save(str(data_path), arr, allow_pickle=False)
  print(f'Saved {str(data_path)}', flush=True)

  len_path: Path = out_shard_path.with_suffix('.len.npy')
  np.save(str(len_path), lengths, allow_pickle=False)
  print(f'Saved {str(len_path)}', flush=True)

def jsonl_to_str(str_samps: Queue, in_shard: Path, consumer_threads: int) -> None:
  print('jsonl_to_str: Running', flush=True)
  with gzip.GzipFile(filename=str(in_shard)) as g:
    for line in g:
      obj: Dict[str, Any] = json.loads(line)
      text: str = obj['text']
      str_samps.put(text)
  for _ in range(consumer_threads):
    str_samps.put(None)
  print('jsonl_to_str: Done', flush=True)

if TYPE_CHECKING:
  from transformers import T5TokenizerFast
  Tokenizer = T5TokenizerFast
else:
  Tokenizer = Any

def str_to_int_worker(str_samps: Queue, int_samps: Queue, tokenizer: Tokenizer) -> None:
  from transformers.utils.generic import TensorType
  from transformers.tokenization_utils_base import BatchEncoding
  from numpy.typing import NDArray
  # print('str_to_int_worker: Running', flush=True)

  def do_task() -> bool:
    samp: Optional[str] = str_samps.get()
    if samp is None:
      return False
    batch: BatchEncoding = tokenizer.__call__(samp, add_special_tokens=False, return_tensors=TensorType.NUMPY, return_attention_mask=False)
    encoded: NDArray = batch['input_ids'][0]
    del batch
    int_samps.put(encoded)
    del encoded
    return True

  while do_task(): pass
  # print('str_to_int_worker: Done', flush=True)

def str_to_int_manager(str_samps: Queue, int_samps: Queue, threads: int) -> None:
  from transformers import T5TokenizerFast
  tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained('google/t5-v1_1-base', legacy=False)
  with ThreadPool(threads) as pool:
    _: List[AsyncResult] = [pool.apply_async(str_to_int_worker, args=(str_samps, int_samps, tokenizer)) for _ in range(threads)]
    pool.close()
    pool.join()
  print('>str_to_int_manager done.', flush=True)

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--in-dir', type=Path, help='directory in which c4-(train|validation).*-of-*.json.gz reside. download from https://huggingface.co/datasets/allenai/c4/tree/main/en')
  p.add_argument('--out-dir', default=Path('out'), type=Path, help='directory into which to output ragged arrays')
  p.add_argument('--consumer-threads', default=1, type=int, help='threads per consumer process')
  p.add_argument('--data-slab-size', default=512*1024**2, type=int, help='bytes to allocate for ragged array data')
  p.add_argument('--length-slab-size', default=2*1024**2, type=int, help='bytes to allocate for ragged array indices')
  args = p.parse_args()

  for split in ('train', 'validation'):
    out_split_dir: Path = args.out_dir / split
    makedirs(str(out_split_dir), exist_ok=True)

    in_shards_unsorted: List[str] = fnmatch.filter(listdir(str(args.in_dir)), f'c4-{split}.*-of-*.json.gz')
    in_shards_total: int = len(in_shards_unsorted)
    matcher = f'^c4-{split}.(\d+)-of-(\d+).json.gz$'
    get_shard_ix: Callable[[str], int] = lambda fname: int(re.search(matcher, fname).group(1))
    in_keyer: Callable[[str], int] = lambda fname: get_shard_ix(Path(fname).name)
    in_shards: List[str] = sorted(in_shards_unsorted, key=in_keyer)

    out_shards_unsorted: List[str] = fnmatch.filter(listdir(str(out_split_dir)), f'c4-{split}.*-of-*.data.npy')
    out_shards_set: Set[str] = set(out_shards_unsorted)

    def convert_shard(in_shard_path: Path, out_shard_path: Path):
      str_samps = Queue(maxsize=128)
      int_samps = Queue(maxsize=128)
      # https://superfastpython.com/threadpool-producer-consumer/
      int_to_file_ = Thread(target=int_to_file, args=(int_samps, out_shard_path, args.data_slab_size, args.length_slab_size))
      int_to_file_.start()
      str_to_int = Thread(target=str_to_int_manager, args=(str_samps, int_samps, args.consumer_threads))
      str_to_int.start()
      jsonl_to_str_ = Thread(target=jsonl_to_str, args=(str_samps, in_shard_path, args.consumer_threads))
      jsonl_to_str_.start()

      str_to_int.join()
      jsonl_to_str_.join()
      int_samps.put(None)
      int_to_file_.join()

    for in_shard in in_shards:
      shard_ix: int = get_shard_ix(in_shard)
      shard_out_name: str = f'c4-{split}.{shard_ix:05d}-of-{in_shards_total:05d}.npy'
      if shard_out_name in out_shards_set:
        continue
      print(f'converting shard {in_shard}...')
      convert_shard(args.in_dir / in_shard, out_split_dir / shard_out_name)
    print('>main done.')