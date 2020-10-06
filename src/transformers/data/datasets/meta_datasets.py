import os
import pickle
import time

import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Sampler

from typing import Dict

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

        self.file_path = file_path

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
            # need to get the attention masks too?

        else:
            logger.debug(f"Creating features from dataset file at {directory}")

            self.examples = []

            # TODO: I should not need the errors here, clean data more to ensure this is fixed.
            with open(file_path, encoding="utf-8", errors="replace") as f:
                text = f.read()

            logger.debug(f"Tokenizing dataset file")

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            # TODO: Is this proper behavior? Do we get training data like this?
            for i in range(0, len(tokenized_text), block_size):
            # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            #     partial_len = None
                if i + block_size <= len(tokenized_text):
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                else:
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i:])
                    )
                    # # tmp = torch.zeros(block_size)
                    # tokens = tokenizer.build_inputs_with_special_tokens(tokenized_text[i:])
                    # # partial_len = len(tokens)
                    # tmp[:partial_len] = torch.tensor(tokens)
                    # self.examples.append(
                    #     tmp
                    # )
                # TODO: must make sure that train and eval batch sizes are the same
                # atn_mask = torch.zeros(block_size)
                # tmp = torch.tensor(self.examples[-1])
                # if partial_len:
                #     atn_mask[:partial_len] = torch.ones(partial_len)
                # else:
                #     atn_mask[:tmp.shape[0]] = torch.ones(tmp.shape[0])
                # self.attention_masks.append(
                #     atn_mask
                # )
            # # TODO: test this - print a few examples and masks
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('the first entry of examples and attention masks')
            # print(self.examples[0].shape, self.attention_masks[0].shape)
            # print(self.examples[0])
            # print(self.attention_masks[0])
            # print('the last entry of examples and attention masks')
            # print(self.examples[-1].shape, self.attention_masks[-1].shape)
            # print(self.examples[-1])
            # print(self.attention_masks[-1])
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # exit(0)

            # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            # Ignore incomplete batches
            # If you don't do this, you'll get an error at the end of training
            # TODO: verify this - I may need this after all, so I can train with very small amounts of data.
            # n = len(self.examples) % train_batch_size
            # if n != 0:
            #     self.examples = self.examples[:-n]

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

    # def get_attention_mask(self, i) -> torch.Tensor:
    #     torch.tensor(self.attention_masks[i], dtype=torch.long)

    def size(self, i) -> int:
        return len(self.examples)

    def get_filepath(self):
        return self.file_path


class GutenburgDataset(Dataset):
    # first get a list of all the files we care about
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            train_batch_size=1,
            eval=False,
            k=2500
    ):
        assert os.path.isdir(file_path)

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.overwrite_cache = overwrite_cache
        self.train_batch_size = train_batch_size
        self.eval = eval
        self.k = k

        # we have a list of books, now when we server each dataset, create the book datasets and return
        logger.info('building Gutenburg dataset')
        self.books = []
        for dirpath, dirnames, filenames in os.walk(file_path):
            for filename in [f for f in filenames if f.endswith('.txt') and 'cached_' not in f]:
                self.books.append(os.path.join(dirpath, filename))
        logger.info(f'Gutenburg dataset built, containing {len(self.books)} books.')

    def __len__(self):
        return len(self.books)

    def __getitem__(self, i) -> Dict[str, Dataset]:
        '''
        Args:
            i:

        Returns:

        '''
        if self.k == 250:
            file_extension = '250'
        elif self.k == 500:
            file_extension = '500'
        elif self.k == 1000:
            file_extension = '1000'
        elif self.k == 2500:
            file_extension = '2500'
        elif self.k == 5000:
            file_extension = '5000'
        elif self.k == 10000:
            file_extension = '10000'
        else:
            assert False, 'the token size is not recognized'

        train_file_extension = '.' + file_extension + 'metatrain'
        test_file_extension = '.metatest'

        if self.eval:
            metatrain = BookDataset(
                tokenizer=self.tokenizer,
                file_path=self.books[i] + train_file_extension,
                block_size=self.block_size,
                overwrite_cache=self.overwrite_cache,
                train_batch_size=self.train_batch_size
            )
            metatest = BookDataset(
                tokenizer=self.tokenizer,
                file_path=self.books[i] + test_file_extension,
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
