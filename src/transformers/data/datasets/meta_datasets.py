import os
import pickle
import time

import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Sampler

import logging

logger = logging.getLogger(__name__)


# TODO: I think I need to build a gutenburg and a bookdataset sampler and collator_fn




class BookDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            train_batch_size=1
    ):
        assert os.path.isfile(file_path), file_path

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.debug(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )

        else:
            logger.debug(f"Creating features from dataset file at {directory}")

            self.examples = []
            # TODO: I should not need the errors here, clean data more to ensure this is fixed.
            with open(file_path, encoding="utf-8", errors="replace") as f:
                text = f.read()

            logger.debug(f"Tokenizing dataset file")

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            # TODO: Is this proper behavior? Do we get training data like this?
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                )

            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            # Ignore incomplete batches
            # If you don't do this, you'll get an error at the end of training
            # TODO: verify this
            n = len(self.examples) % train_batch_size
            if n != 0:
                self.examples = self.examples[:-n]

            logger.debug(f"The number of examples for this dataset file is: {len(self.examples)}")

            start = time.time()
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(
                f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

    def size(self, i) -> int:
        return len(self.examples)


class GutenburgDataset(Dataset):
    # first get a list of all the files we care about
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            train_batch_size=1,
            eval=False
    ):
        assert os.path.isdir(file_path)

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache
        self.train_batch_size = train_batch_size
        self.eval = eval

        # we have a list of books, now when we server each dataset, create the book datasets and return
        self.books = []
        for dirpath, dirnames, filenames in os.walk(file_path):
            for filename in [f for f in filenames if f.endswith('.txt') and 'cached_' not in f]:
                self.books.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.books)

    def __getitem__(self, i) -> Dataset:
        '''
        Args:
            i:

        Returns:

        '''
        # I Don't think I need this - or maybe I do but just for eval as is noted.
        if self.eval:
            metatrain = BookDataset(
                tokenizer=self.tokenizer,
                file_path=self.books[i]+'.metatrain',
                block_size=self.block_size,
                overwrite_cache=self.overwrite_cache,
                train_batch_size=self.train_batch_size
            )
            metatest = BookDataset(
                tokenizer=self.tokenizer,
                file_path=self.books[i] + '.metatest',
                block_size=self.block_size,
                overwrite_cache=self.overwrite_cache,
                train_batch_size=self.train_batch_size
            )
            # TODO: I don't know how to handle this
            # This needs to return a dataset, but at the same time, I need both datasets
            return {'metatrain': metatrain, 'metatest': metatest}

        else:
            bookdataset = BookDataset(
                tokenizer=self.tokenizer,
                file_path=self.books[i],
                block_size=self.block_size,
                overwrite_cache=self.overwrite_cache,
                train_batch_size=self.train_batch_size
            )
            return bookdataset


# class GutenburgSampler(Sampler):
#     r"""
#
#     """
#
#     def __init__(self, data_source):
#         assert isinstance(data_source, GutenburgDataset), 'This sampler can only be used with a Gutenburg Dataset'
#
#         self.data_source = data_source
#
#     @property
#     def num_samples(self):
#         return len(self.data_source)
#
#     def __iter__(self):
#         n = len(self.data_source)
#         return iter(torch.randperm(n, generator=self.generator).tolist())
#
#     def __len__(self):
#         return self.num_samples
