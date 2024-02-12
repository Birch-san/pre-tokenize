from time import sleep
from random import random
from pathlib import Path
from multiprocessing import Process, Queue #JoinableQueue
from multiprocessing.pool import ThreadPool, AsyncResult
import fnmatch
from os import makedirs, listdir
import argparse
from typing import List, Callable, Set, Generator, BinaryIO, Dict, Any
import re
import numpy as np
from numpy.typing import NDArray
import io
from threading import Thread
from io import BytesIO
import gzip
import json

# https://stackoverflow.com/a/12572031/5257399
def stream_gzip(stream: BinaryIO) -> Generator[bytes, None, None]:
  import zlib
  dec = zlib.decompressobj(32 + zlib.MAX_WBITS)
  for chunk in stream:
    rv = dec.decompress(chunk)
    if rv:
      yield rv
  if dec.unused_data:
    # decompress and yield the remainder
    yield dec.flush()

def producer(queue: Queue, in_shard: Path) -> None:
  import ijson.backends.yajl2_cffi as ijson
  # jb = ijson.get_backend('yajl2_c')
  # buffer_size=4096
  # with io.open(str(in_shard), mode='rb', buffering=buffer_size, errors=None, closefd=True) as f:
  with gzip.GzipFile(filename=str(in_shard)) as g:
    for line in g:
      obj: Dict[str, Any] = json.loads(line)
      text: str = obj['text']
      queue.put(text)
    # while g.readline()
    # gstream: Iterable[bytes] = stream_gzip(f, buffer_size=buffer_size)
    # BytesIO
    # gzip.GzipFile(fileobj=io.BytesIO(compressed_data))
    # g.read
      # tups = ijson.basic_parse(g)
      # parsed = ijson.items(tups, prefix='text')
      # n = next(parsed)
    pass
  # import dask
  # dask.config.set({'dataframe.query-planning': True})
  # import dask.dataframe as dd
  # from dask.dataframe.core import DataFrame
  # df: DataFrame = dd.read_json(str(in_shard), lines=True, orient='records', compression='gzip')
  # for samp in df.text:
  #   queue.put(samp)
  queue.put(None)
  # queue.task_done()

def consumer_task(queue: Queue) -> None:
  print('Consumer: Running', flush=True)
  while True:
    item = queue.get()
    if item is None:
      break
    print(f'>got {item}', flush=True)
  print('Consumer: Done', flush=True)

def consumer_manager(queue: Queue, threads: int) -> None:
  with ThreadPool(threads) as pool:
    _: List[AsyncResult] = [pool.apply_async(consumer_task, args=(queue,)) for _ in range(threads)]
    pool.close()
    pool.join()
  print('>consumer_manager done.')

if __name__ == '__main__':
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument('--in-dir', type=Path, help='directory in which c4-(train|test).*-of-*.json.gz reside. download from https://huggingface.co/datasets/allenai/c4/tree/main/en')
  p.add_argument('--out-dir', default=Path('out'), type=Path, help='directory into which to output ragged arrays')
  p.add_argument('--consumer-threads', default=1, type=int, help='threads per consumer process')
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

    out_shards_unsorted: List[str] = fnmatch.filter(listdir(str(out_split_dir)), f'c4-{split}.*-of-*.npy')
    out_shards_set: Set[str] = set(out_shards_unsorted)

    for in_shard in in_shards:
      shard_ix: int = get_shard_ix(in_shard)
      shard_out_name: str = f'c4-{split}.{shard_ix:05d}-of-{in_shards_total:05d}.npy'
      if shard_out_name in out_shards_set:
        pass
      str_samps = Queue()
      # https://superfastpython.com/threadpool-producer-consumer/
      consumer_ = Thread(target=consumer_manager, args=(str_samps, args.consumer_threads))
      consumer_.start()
      producer_ = Thread(target=producer, args=(str_samps, args.in_dir / in_shard,))
      producer_.start()
      producer_.join()
      consumer_.join()
      pass
    print('>main done.')