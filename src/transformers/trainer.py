import json
# import logging
import math
import os
import random
import re
import shutil
import copy
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from .data.datasets.meta_datasets import BookDataset
from .data.data_collator import DataCollator, DefaultDataCollator, DataCollatorForMetaLanguageModeling
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from .training_args import TrainingArguments, is_tpu_available
from .data import Logger

import threading

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False

try:
    import mlflow
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


# logging.setLoggerClass(Logger)
# logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        logger: Logger = None,

    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.logger = logger
        if data_collator is not None:
            self.data_collator = data_collator
        elif self.args.meta != 'none':
            self.data_collator = DataCollatorForMetaLanguageModeling()
        else:
            self.data_collator = DefaultDataCollator()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            self.logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            self.logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        self.optimizers = optimizer, scheduler
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        self.logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        # for use in saving model to backend MLFlow
        savethread = None

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        self.logger.info("  Num Epochs = %d", num_train_epochs)
        self.logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                self.logger.info("  Continuing training from epoch %d", epochs_trained)
                self.logger.info("  Continuing training from global step %d", self.global_step)
                self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        self.logger.save_model(model,
                                               optimizer,
                                               'checkpoint' + str(self.global_step),
                                               logging_loss,
                                               self.global_step)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            # maybe save model here? once Every epoch? Weird that it doesn't correspond to the training though?

        if self.tb_writer:
            self.tb_writer.close()

        self.logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        this_loss = loss.item()
        if self.global_step % 25 == 0:
            self.logger.log_train(self.global_step, this_loss)

        return this_loss

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            self.logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        # TODO: this is the generic evaluation method.  I will modify this to my own purposes,
        #  And then just like in the training example,I'll use the _prediction loop to cover any differences
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        self.logger.info("***** Running %s *****", description)
        self.logger.info("  Num examples = %d", self.num_examples(dataloader))
        self.logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = logits.detach()
                else:
                    preds = torch.cat((preds, logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output


class MetaTrainer(Trainer):

    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            prediction_loss_only=False,
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            inner_collator: Optional[DataCollator] = None,
            logger: Logger = None,
    ):
        """
        MetaTrainer extends the transformer trainer, to allow for meta-capabilities (but only first-order at this point)
        """
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, compute_metrics,
                         prediction_loss_only, tb_writer, optimizers, logger)
        self.inner_collator = inner_collator
        self.eval_collator = inner_collator if inner_collator is not None else data_collator

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        # if not metalearning or conditioning, just use the original training method
        if self.args.meta == 'none':
            return super(MetaTrainer, self).train(model_path)

        # in this case, the dataloader length is the number of books
        train_dataloader = self.get_train_dataloader()

        # TODO: this calculation is misleading - the len of the dataloader isn't very meaningful.
        #  what we want is the len of all of the metatest sets of the dataloader
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        # for use in saving model to backend MLFlow
        savethread = None

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        self.logger.info("  Num Epochs = %d", num_train_epochs)
        self.logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        self.logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                self.logger.info("  Continuing training from epoch %d", epochs_trained)
                self.logger.info("  Continuing training from global step %d", self.global_step)
                self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        meta_gradients = []

        # this is probably fine, regardless of how we track the epochs, one epoch is still one epoch
        for epoch in train_iterator:

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            for step, datasets in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                # datasets is a batch of datasets.  For each dataset in the batch, get the loss on metatest of \theta'
                # then update once for the batch
                for dset in datasets:

                    metatest_set = dset[0]['metatest']
                    metatest_sampler = RandomSampler(metatest_set)
                    metatest_loader = DataLoader(metatest_set,
                                                 batch_size=self.args.train_batch_size,  # TODO, is this right?
                                                 sampler=metatest_sampler,
                                                 collate_fn=self.inner_collator.collate_batch)  # TODO, is this right?

                    # build the metatrain dataloader
                    metatrain_set = dset[0]['metatrain']
                    metatrain_sampler = RandomSampler(metatrain_set)
                    metatrain_loader = DataLoader(metatrain_set,
                                                  batch_size=self.args.train_batch_size,  # TODO, is this right?
                                                  sampler=metatrain_sampler,
                                                  collate_fn=self.inner_collator.collate_batch)  # TODO: is this right?

                    # actually, we probably shouldn't just train on all the info of one book. Better idea is to train on
                    # only one, or however many we need for a batch
                    for stp, inputs in enumerate(metatest_loader):

                        # for every batch in the metatest set
                        tr_loss += self._training_step(model, inputs, optimizer, metatrain_loader)

                        # keep track of gradients, want them all
                        # now I have the gradients, but we want to accumulate them for all of this loop.
                        for idx, p in enumerate(model.parameters()):
                            if len(meta_gradients) < 1:
                                meta_gradients = [float(0) for _ in model.parameters()]
                            meta_gradients[idx] += p.grad / len(metatest_loader)

                        # if we are to update on one batch, then we only need one sample from each book
                        break

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):

                    # now ensure the gradients are set properly
                    for g, p in zip(meta_gradients, model.parameters()):
                        p.grad = g

                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    # clear gradeints
                    meta_gradients = []

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                        if self.args.evaluate_during_training:
                            self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        self.logger.save_model(model,
                                               optimizer,
                                               'checkpoint' + str(self.global_step),
                                               logging_loss,
                                               self.global_step)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            # maybe save model here? once Every epoch? Weird that it doesn't correspond to the training though?

        if self.tb_writer:
            self.tb_writer.close()

        self.logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _training_step(
        self, model: nn.Module, inputs: BookDataset, optimizer: torch.optim.Optimizer,
            metatrain_loader: Optional[DataLoader] = None
    ) -> float:

        if self.args.meta == 'fomaml' or self.args.meta == 'conditioning':

            if not metatrain_loader:
                raise ValueError('metatrain_loader must not be None')

            return self._fomaml_training_step(model, inputs, optimizer, metatrain_loader)

        elif self.args.meta == 'none':
            return super(MetaTrainer, self)._training_step(model, inputs, optimizer)

        else:
            raise NotImplementedError('the requested meta-learning method is not implemented.')


    def _fomaml_training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
            metatrain_loader: Optional[DataLoader]
    ) -> float:
        '''
        I think this is wrong.  What this training step should take in is a metatest set, and a metatrain set.
        Then this function should take num_inner_steps updates on the metatrain set, and then return the gradient
        calculated on the metatest set.

        '''

        '''
        So here is what I can do:
        leave trainier the same, and in this case we can think of the gradient accumulation steps as the meta-batch
        size. So in the outer step, we will sample a task (i.e. a book), and then pass this into the training_step.
        In the training step, we will duplicate the model parameters, then update the original parameters,
        sample another datapoint, then calculate the gradient of this, and then assign this gradient to the original
        model parameters.

        So what does this look like with multiple inner steps? With multiple inner steps, we simply repeat the
        "sample another datapoint, calculate the gradient of this" multiple times.

        Exactly what the mechanism of duplicating, updating, and assigning gradients to parameters is, I'm not sure.
        '''
        conditioning = False
        if self.args.meta == 'conditioning':
            conditioning = True

        model.train()
        # inputs is a list of book datasets, so I should just train a model for each of those, given a few steps
        if not conditioning:
            original_parameters = copy.deepcopy(model.state_dict())

        '''
        inputs here can be one of two things:
        1) a dict of ['metatrain': metatrain_dataset, 'metatest': metatest_dataset]
        2) must the meta_train dataset. 
        
        overall, in this method, we want one outer step to be taken.  
        Therefore, we must do however many inner updates on the metatrain set. Then, 
        given the network that we have ended with, calculate loss on the metatest set, and update the original model.  
        '''
        inner_step = 0
        done_training = False
        cond_tokens = None

        while not done_training:

            for train_input in metatrain_loader:

                # for this book, update and then train and shit
                for k, v in train_input.items():
                    train_input[k] = v.to(self.args.device)
                    if conditioning:
                        cond_tokens = v[0][:self.args.k].unsqueeze(0)
                        break

                if conditioning:
                    done_training = True
                    break

                outputs = model(**train_input)

                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                model.zero_grad()

                if inner_step >= self.args.num_inner_steps * self.args.gradient_accumulation_steps:
                    # we are finished training this model, so we need to take one more step
                    # (I think, and just save the gradient)
                    done_training = True
                    break

                inner_step += 1

        # now get gradients for new model on the given input data.
        # Then set the gradient of the original model appropriately
        # add the conditioning tokens to v
        for k, v in inputs.items():
            if conditioning:
                inputs[k] = torch.cat((cond_tokens.repeat(v.shape[0], 1), v), 1).to(self.args.device)
            else:
                inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # save the metatest gradients (i.e. the outer step gradients)
        if not conditioning:
            gradients = []
            for param in model.parameters():
                gradients.append(param.grad)

            # restore the orginal model parameters
            if not conditioning:
                model.load_state_dict(original_parameters)

            # apply the meta gradient to the original network
            for p, g in zip(model.parameters(), gradients):
                p.grad = g
        # only log every 10 steps
        if self.global_step % 10 == 0:
            self.logger.log_train(self.global_step, loss.item())

        return loss.item()

    def _prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        if self.args.meta == 'none' or self.args.meta == 'fomaml':
            return self._standard_prediction_loop(dataloader, description, prediction_loss_only)
        elif self.args.meta == 'conditioning':
            return self._standard_prediction_loop(dataloader, description, prediction_loss_only, condition=True)
        else:
            raise NotImplementedError('the requested meta-learning method is not implemented.')

    def _standard_prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None,
            condition: Optional[bool] = False,
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        book_performances = []

        batch_size = dataloader.batch_size
        self.logger.info("***** Running %s *****", description)
        self.logger.info("  Num examples = %d", self.num_examples(dataloader))
        self.logger.info("  Batch size = %d", batch_size)

        # model.eval()
        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        original_model = copy.deepcopy(model.state_dict())
        model.zero_grad()

        # get an optimizer if we don't have one already
        if self.optimizers is not None:
            optimizer, scheduler = self.optimizers
        else:
            optimizer, scheduler = self.get_optimizers(self.args.num_eval_finetune_steps)

        # instead of using the dataloader, just use the dataset
        for bookdset in tqdm(dataloader.dataset, desc=description):

            bookdset = bookdset[0]
            # is bookdset a dict of metatrain metatest?
            # make sure the model is set as the original paramers
            model.train()

            model.load_state_dict(copy.deepcopy(original_model))

            eval_losses: List[float] = []
            preds: torch.Tensor = None
            label_ids: torch.Tensor = None
            skip_training = False
            if len(bookdset['metatrain']) < 1:
                # This may happen if there is not a full batch, so just skip to the eval part
                # (This is a problem for the smaller token counts)
                print('the metatrain dataset is empty here, continuing.')
                tmp = bookdset['metatrain'].get_filepath()
                print(f"filename: {tmp}")
                skip_training = True

            if not skip_training:
                train_sampler = RandomSampler(bookdset['metatrain'])
                train_loader = DataLoader(bookdset['metatrain'],
                                          batch_size=self.args.train_batch_size,
                                          sampler=train_sampler,
                                          collate_fn=self.inner_collator.collate_batch)

                # do the fine-tuning step
                inner_step_done = False
                trained_steps = 0

                # if conditioning, inner step is done
                if self.args.meta == 'conditioning':
                    inner_step_done = True

                while not inner_step_done:
                    for inner_step, book_data in enumerate(train_loader):
                        # for this book, update and then train and shit
                        if trained_steps >= self.args.num_eval_finetune_steps * self.args.gradient_accumulation_steps:
                            inner_step_done = True
                            break

                        trained_steps += 1

                        for k, v in book_data.items():
                            # keys are input_ids, labels, pad information if necessary
                            book_data[k] = v.to(self.args.device)

                        # pad book_data if necessary


                        outputs = model(**book_data)
                        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                        if self.args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu parallel training
                        if self.args.gradient_accumulation_steps > 1:
                            loss = loss / self.args.gradient_accumulation_steps

                        if self.args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        optimizer.step()
                        model.zero_grad()

            test_sampler = SequentialSampler(bookdset['metatest'])
            test_loader = DataLoader(bookdset['metatest'],
                                     batch_size=self.args.eval_batch_size,
                                     sampler=test_sampler,
                                     collate_fn=self.inner_collator.collate_batch)  # TODO: verify this eval_collator

            # put model in eval mode
            model.eval()

            # set the conditioning tokens
            # not sure exactly how this should be done
            if skip_training:
                assert not condition, 'training was skipped due to no metatrain data, but we are conditioning. '
            if condition:
                for _, cond_data in enumerate(train_loader):
                    cond_tokens = cond_data['input_ids'][0][:self.args.k].unsqueeze(0)
                    break

            # now eval on the test set
            # print(f'the test loader len is (during eval) is: {len(test_loader)}')
            for inputs in test_loader:
                has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

                for k, v in inputs.items():
                    if condition:
                        inputs[k] = torch.cat((cond_tokens.repeat(v.shape[0], 1), v), 1).to(self.args.device)
                    else:
                        inputs[k] = v.to(self.args.device)

                with torch.no_grad():
                    # if conditioning, then we need to run multiple steps here
                    # if condition:
                    #     # build the new inputs - should be of same length, just more of them?
                    
                    outputs = model(**inputs)
                    if has_labels:
                        step_eval_loss, logits = outputs[:2]
                        # if conditioning, only count the loss of the non conditioning tokens
                        if condition:
                            loss_fn = torch.nn.CrossEntropyLoss()
                            logits = logits[..., :-1, :].contiguous()
                            targets = inputs['labels'][..., 1:].contiguous()
                            logits = logits[:, self.args.k:, :].contiguous()
                            targets = targets[:, self.args.k:].contiguous()
                            logits = logits.view(-1, logits.size(-1))
                            targets = targets.view(-1)
                            if len(targets) < 1:
                                continue
                            loss = loss_fn(logits, targets)
                            eval_losses += [loss.mean().item()]
                            # eval_losses += [step_eval_loss.mean().item()]
                        else:
                            eval_losses += [step_eval_loss.mean().item()]
                    else:
                        logits = outputs[0]
                        assert False, "I don't think that this works.  Must use labels."

                if not prediction_loss_only:
                    if preds is None:
                        preds = logits.detach()
                    else:
                        preds = torch.cat((preds, logits.detach()), dim=0)
                    if inputs.get("labels") is not None:
                        if label_ids is None:
                            label_ids = inputs["labels"].detach()
                        else:
                            label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

            if self.args.local_rank != -1:
                # In distributed mode, concatenate all results from all nodes:
                if preds is not None:
                    preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
                if label_ids is not None:
                    label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
            elif is_tpu_available():
                # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
                if preds is not None:
                    preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
                if label_ids is not None:
                    label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

            # Finally, turn the aggregated tensors into numpy arrays.
            if preds is not None:
                preds = preds.cpu().numpy()
            if label_ids is not None:
                label_ids = label_ids.cpu().numpy()

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            else:
                metrics = {}
            #np.exp(eval_losses)
            if len(eval_losses) > 0:
                metrics["perplexity"] = np.mean(np.exp(eval_losses))
                metrics["loss"] = np.mean(eval_losses)

                # Prefix all keys with eval_
            for key in list(metrics.keys()):
                if not key.startswith("eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)

            book_performances.append((PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics), bookdset['metatest'].get_filepath()))
            # now, I need so somehow transfer all of this into a performance conglomerate

        all_losses = []
        all_perplex = []
        all_filepaths = []
        for bp in book_performances:
            all_losses.append(bp[0].metrics['eval_loss'])
            all_perplex.append(bp[0].metrics['eval_perplexity'])
            all_filepaths.append(bp[1])
        avg_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)
        ste_loss = std_loss / math.sqrt(len(all_losses))
        avg_perplex = np.mean(all_perplex)
        std_perplex = np.std(all_perplex)
        ste_perplex = std_perplex / math.sqrt(len(all_perplex))

        self.logger.log_eval(avg_loss, avg_perplex, self.global_step, std_loss, std_perplex, ste_loss, ste_perplex)
        stp = self.global_step if self.global_step is not None else 0
        self.logger.log_json({'all_perplex': all_perplex, 'step': stp, 'files': all_filepaths},
                             self.args.run_name + '_all_perplexities_' + str(stp) + '.json')
        print('perp:\tfilepaths')
        for perp, fp in zip(all_perplex, all_filepaths):
            print(f'{perp}:\t{fp}')
        return book_performances[-1][0]  # This is meaningless, meaning the _log is meaningless, just use MLFlow


