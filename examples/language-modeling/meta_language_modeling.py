# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


# TODO: Make sure that the evaluation metric is not tied up in the perplexity measurement
# TODO: Implement a meta-learning version of this

import logging
import math
import os
import copy
from random import shuffle
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForMetaLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    MetaTrainer,
    TrainingArguments,
    set_seed,
    Logger,
    GutenburgDataset,
    BookDataset,
    TextDataset
)

# from dataloaders.text_dataset import TextDataset
import mlflow
# from dataloaders.gutenburg_dataset import GutenburgDataset

import numpy as np

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    run_name: str = field(
        default='testing', metadata={"help": "What name to store the current run under in the MLFlow backend"}
    )

    training_epochs: int = field(
        default=3, metadata={"help": "how many epochs to train on the training set for."}
    )

    finetune_epochs: int = field(
        default=0, metadata={"help": "How many epoch to finetune for"}
    )

    meta: str = field(
        default='none', metadata={"help": "Which meta-learning method to use, defaults to None"}
    )

    num_inner_steps: int = field(
        default=1, metadata={"help": "the number of inner steps to take for meta-learning"}
    )

    num_eval_finetune_steps: int = field(
        default=0, metadata={"help": "The number of steps to take on the meta-fine-tune set during evaluation"}
    )

    k: int = field(
        default=2500, metadata={"help": "The number of tokens to use in the meta-train sets"}
    )

    drop_incomplete_blocks: bool = field(
        default=False, metadata={"help": "Set to true to drop tokenized blocks with length shorter than block_size"}
    )

    keep_all_in_memory: bool = field(
        default=False, metadata={"help": "Set to true to keep all data in memory"}
    )

    use_all_for_training: bool = field(
        default=False, metadata={"help": "If set to true, then data from the meta-train and meta-test sets of the "
                                         "training set will be used for the inner loop during training."}
    )


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    block_size = args.block_size - args.k if args.meta == 'conditioning' else args.block_size
    if args.meta == "none" and not evaluate:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=block_size
        )
    else:
        return GutenburgDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size,
                                train_batch_size=training_args.per_gpu_train_batch_size * training_args.n_gpu,
                                k=data_args.k, keep_all_in_memory=data_args.keep_all_in_memory)


def main(model_args, data_args, training_args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns

    # For now, just make meta-variable available in training_args
    training_args.meta = data_args.meta
    training_args.num_inner_steps = data_args.num_inner_steps
    training_args.num_eval_finetune_steps = data_args.num_eval_finetune_steps
    training_args.k = data_args.k
    training_args.run_name = data_args.run_name
    training_args.use_all_for_training = data_args.use_all_for_training

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir, return_attention_mask=True)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, return_attention_mask=True)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, return_attention_mask=True)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, return_attention_mask=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    # enable padding so we can use small amounts of (meta) training data
    tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # num_params = sum(p.numel() for p in model.parameters())
    # print(f'the number of parameters in the model is: {num_params}')

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    logger.info('creating data collator')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability, block_size=data_args.block_size
    )

    # TODO: is this correct for conditioning?
    if data_args.meta != 'none':
        outer_collator = DataCollatorForMetaLanguageModeling()
    logger.info('data collator created')

    logger.info('initializing trainer')
    # Initialize our Trainer
    trainer = MetaTrainer(
        model=model,
        args=training_args,
        data_collator=outer_collator if training_args.meta != 'none' else data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        inner_collator=data_collator,
        logger=logger,
    )
    logger.info('trainer initialized')

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        train_output = trainer.train(model_path=model_path)

        # this logging is questionable
        logger.save_model(trainer.model,
                          trainer.optimizers,
                          os.path.join(training_args.output_dir, 'model_bundle-' + str(train_output.global_step) + '.mdl'),
                          train_output.training_loss,
                          train_output.global_step)

        # TODO: This is slow, must trasfer 1.5G over network
        logger.info('saving trainged model')
        trainer.save_model()
        # TODO: do I need to save the tokenizer as well?
        logger.info('saving pretrained tokenizer, but only locally')
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    mlflow.set_tracking_uri('http://gs18196.sp.cs.cmu.edu:6460')
    mlflow.set_experiment("guten")
    with mlflow.start_run(run_name=data_args.run_name):
        all_args = {}
        for d in [model_args, data_args, training_args]:
            for key, val in vars(d).items():
                all_args[key] = val

        logger = Logger(__name__, all_args)
        # print(mlflow.get_tracking_uri())
        # print(mlflow.get_artifact_uri())
        main(model_args, data_args, training_args)
