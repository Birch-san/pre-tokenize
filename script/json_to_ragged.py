from pathlib import Path
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool, AsyncResult
import fnmatch
from os import makedirs, listdir
import argparse
from typing import List, Callable, Set, Dict, Any
import re
from threading import Thread
import gzip
import json

def producer(queue: Queue, in_shard: Path) -> None:
  with gzip.GzipFile(filename=str(in_shard)) as g:
    for line in g:
      obj: Dict[str, Any] = json.loads(line)
      text: str = obj['text']
      queue.put(text)
  queue.put(None)

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