import os
import pickle
import time

import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Sampler

from typing import Dict, List

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
            overwrite_cache: bool = False,
            drop_incomplete: bool = False
    ):
        assert os.path.isfile(file_path), file_path

        self.file_path = file_path

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        ic = '_nic' if drop_incomplete else ''
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}{}_{}".format(tokenizer.__class__.__name__, str(block_size), ic, filename,),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            start = time.time()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)

            logger.debug(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
            # need to get the attention masks too?

        else:
            logger.debug(f"Creating features from dataset file at {directory}")

            self.examples = []

            with open(file_path, encoding="utf-8", errors="replace") as f:
                text = f.read()

            logger.debug(f"Tokenizing dataset file")

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text), block_size):
                if i + block_size <= len(tokenized_text):
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                else:
                    if not drop_incomplete:
                        self.examples.append(
                            tokenizer.build_inputs_with_special_tokens(tokenized_text[i:])
                        )

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

    def get_filepath(self):
        return self.file_path


class GutenburgDataset(Dataset):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache: bool = False,
            train_batch_size: int = 1,
            k: int = 2500,
            drop_incomplete: bool = False,
            keep_all_in_memory: bool = False
    ):
        assert os.path.isdir(file_path)

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache
        self.train_batch_size = train_batch_size
        self.eval = eval
        self.k = k
        self.drop_incomplete = drop_incomplete
        self.keep_all_in_memory = keep_all_in_memory
        self.file_extensions = dict([
            (250, '250'),
            (500, '500'),
            (1000, '1000'),
            (2500, '2500'),
            (5000, '5000'),
            (10000, '10000'),
        ])

        # we have a list of books, now when we server each dataset, create the book datasets and return
        logger.info('building Gutenburg dataset')
        self.books = []
        for dirpath, dirnames, filenames in os.walk(file_path):
            for filename in [f for f in filenames if f.endswith('.txt') and 'cached_' not in f]:
                self.books.append(os.path.join(dirpath, filename))

        self.books_in_mem = []
        if self.keep_all_in_memory:
            # build the book set
            for i in range(len(self.books)):
                metatrain, metatest = self.book_getter(i)
                self.books_in_mem.append((metatrain, metatest))
        
        logger.info(f'Gutenburg dataset built, containing {len(self.books)} books.')

    def __len__(self):
        return len(self.books)

    def __getitem__(self, i: int) -> List[Dict[str, Dataset]]:

        if self.keep_all_in_memory:
            return [{'metatrain': self.books_in_mem[i][0], 'metatest': self.books_in_mem[i][1]}]

        metatrain, metatest = self.book_getter(i)
        return [{'metatrain': metatrain, 'metatest': metatest}]

    def book_getter(self, i):
        if self.k in self.file_extensions.keys():
            file_extension = self.file_extensions[self.k]
        else:
            raise ValueError("GutenburgDataset: invalid value for k")

        train_file_extension = '.' + file_extension + 'metatrain'
        test_file_extension = '.metatest'

        metatrain = BookDataset(
            tokenizer=self.tokenizer,
            file_path=self.books[i] + train_file_extension,
            block_size=self.block_size,
            overwrite_cache=self.overwrite_cache,
            drop_incomplete=self.drop_incomplete
        )
        metatest = BookDataset(
            tokenizer=self.tokenizer,
            file_path=self.books[i] + test_file_extension,
            block_size=self.block_size,
            overwrite_cache=self.overwrite_cache,
            drop_incomplete=False  # never drop the incomplete batch for testing, gotta test on everything
        )

        return metatrain, metatest