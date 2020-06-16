import logging
import os
import pickle
import time
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

import h5py
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in tqdm(range(0, len(tokenized_text) - block_size + 1, block_size)):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


def prepare_line(text, tokenizer, block_size):
    examples = []

    tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        examples.append(
            tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
        )
    if len(tokenized_text) < block_size:
        examples.append(
            tokenizer.build_inputs_with_special_tokens(tokenized_text)
            + [tokenizer.pad_token_id] * (block_size - len(tokenized_text))
        )
    # semaphore.acquire()
    return examples


def maybe_resize(examples: h5py.Dataset, index: int, resize_chunk: int):
    if index >= examples.shape[0]:
        current_shape = list(examples.shape)
        current_shape[0] += resize_chunk
        examples.resize(current_shape)


class ParallelTextDataset(Dataset):
    def __init__(
            self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)
        self.cache_file = None
        self.examples = None

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename, ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.cache_file = h5py.File(cached_features_file, 'r')
                self.examples = self.cache_file['data']
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")
                start = time.time()

                insert_index = 0
                n_jobs = cpu_count()
                with h5py.File(cached_features_file, 'w') as cache_file, ProcessPoolExecutor(n_jobs) as pool, open(file_path) as f:
                    data_examples = cache_file.create_dataset('data', (100, block_size), maxshape=(None, block_size), dtype='i')
                    block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
                    prepare_line_partial = partial(prepare_line, tokenizer=tokenizer, block_size=block_size)
                    for examples in tqdm(pool.map(prepare_line_partial, f, chunksize=n_jobs)):
                        for example in examples:
                            maybe_resize(data_examples, insert_index, 100)
                            data_examples[insert_index] = example
                            insert_index += 1
                self.cache_file = h5py.File(cached_features_file, 'w')
                self.examples = self.cache_file['data']
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


