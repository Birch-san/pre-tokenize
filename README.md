# Pre-tokenizer

The plan is to tokenize the C4 dataset.

Probably we'll start by outputting it as a ragged array (to be space-efficient), and perhaps upload that somewhere.  
Then output it as a 2D numpy array, to be fixed-width and simplify random access. Ideally sparse (zero-padded) or run-length encoded, for space-efficiency. Otherwise we'll have to put up with its taking more space.

We can consider computing some packings using [GraphCore's packedBERT](https://github.com/graphcore/tutorials/tree/sdk-release-2.1/blogs_code/packedBERT), and consider outputting a packed dataset.

## Repository setup

```bash
git clone https://github.com/Birch-san/pre-tokenize.git
cd pre-tokenize
```

### Create & activate virtual environment

```bash
python3 -m venv venv
. venv/bin/activate
```

### Install dependencies

Ensure virtual environment is activated, then:

```bash
pip install -r requirements.txt
```

## Getting C4

We'll download the "en" subset of [c4](https://huggingface.co/datasets/c4) from [`allenai/c4`](https://huggingface.co/datasets/allenai/c4/tree/main/en):

```bash
pip install hf_transfer huggingface-cli
HF_HUB_ENABLE_HF_TRANSFER=True huggingface-cli download --repo-type dataset --cache-dir /sdb/hf-cache --local-dir-use-symlinks True --local-dir /sdb/ml-data/c4 allenai/c4 --include 'en/*'
```

It's 1024 `.json.gz` files, in JSONL format.

## Outputting it as a tokenized, ragged array

We could read the `.json.gz` [using Dask](https://huggingface.co/datasets/allenai/c4#using-dask), then use a [multiprocessing queue](https://superfastpython.com/multiprocessing-queue-in-python/) to tokenize samples.

We should do this a `.json.gz` shard at a time and support `--incremental`, resuming from the last-completed file.

```bash
python -m script.json_to_ragged
```