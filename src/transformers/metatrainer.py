from .trainer import *
from torchmeta.modules.parallel import DataParallel
from torchmeta.utils import gradient_update_parameters
from collections import OrderedDict

try:
    import mlflow
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


def is_mlflow_available():
    return _has_mlflow


FIRST_ORDER_METHODS = ['fomaml']

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
        meta_method: str = 'fomaml',
        finetune_epochs: int = 0,
        num_inner_steps: int = 1,
        first_order: bool = True,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, compute_metrics,
                         prediction_loss_only, tb_writer, optimizers, finetune_epochs, num_inner_steps, first_order)
        self.meta_method = meta_method
        if meta_method in FIRST_ORDER_METHODS:
            self.first_order = True
        else:
            self.first_order = False

    def get_meta_train_dataloader(self) -> DataLoader:
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
            batch_size=1,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
            drop_last=True,
        )

        return data_loader

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """

        # this doesn't support gradient accumulation at the moment
        assert self.args.gradient_accumulation_steps == 1, "gradient accumulation is not supported"

        # TODO: update this method for meta-learning
        train_dataloader = self.get_meta_train_dataloader()
        # TODO: This is needed to build the optimizer schedule
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

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
            # this dataparallel needs to be meta-capable
            model = DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            logging.error("DistributedDataParallel is not supported for metalearning at the moment")

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
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

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

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        step = 0
        for epoch in train_iterator:

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(self.train_dataset, desc="Iteration", disable=not self.is_local_master())

            outer_loss = 0
            # TODO: this will not work because dataloader returns things in terms of batch sizes.
            # TODO: what we want is a dataset that can mix and match books, and then return the dataset there
            # Can we fit the entire dataset in memory?
            for i, bookdataset in enumerate(epoch_iterator):
                assert isinstance(bookdataset, Dataset), 'this is a meta-learning method, so the train_dataloader must return another dataset'

                # now we need to get a dataloader from this dataset
                if is_tpu_available():
                    train_sampler = get_tpu_sampler(self.train_dataset)
                else:
                    train_sampler = (
                        RandomSampler(self.train_dataset)
                        if self.args.local_rank == -1
                        else DistributedSampler(self.train_dataset)
                    )
                bookdataloader = DataLoader(
                    bookdataset,
                    batch_size=self.args.train_batch_size,
                    sampler=train_sampler,
                    collate_fn=self.data_collator.collate_batch,
                    drop_last=True,
                )

                # if step == len(epoch_iterator) - 1:
                #     continue

                # now the book dataset should return a meta-train and meta-test set
                # I had this wrong but it was working?
                # TODO: This doesn't work with the dataloader but does with the dataset
                inner_step = True
                for j, data in enumerate(bookdataloader):
                    step += 1
                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    # TODO: what about gradient accumulation?  For now assume none
                    # gotta do some weird stuff to make inner outer loop work
                    if inner_step:
                        in_loss, params = self._inner_training_step(model, data, epoch * len(epoch_iterator) + step)
                        inner_step = False
                    else:
                        step_loss = self._outer_training_step(model, params, data, optimizer)
                        outer_loss += step_loss
                        inner_step = True

                    if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                    ):
                        if self.args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                        else:
                            tmp = model.parameters()
                            for t in tmp:
                                print(t)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        if is_tpu_available():
                            xm.optimizer_step(optimizer)
                        else:
                            optimizer.step()
                        # log to mlflow too
                        if is_mlflow_available():
                            # personally, I only want this logged every 10 steps or so
                            log_step = epoch * len(epoch_iterator) + step
                            if log_step % 10 == 0:
                                mlflow.log_metric('training/loss', step_loss, log_step)

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
                            # TODO: could add ml flow logging here
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

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _inner_training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], outer_step: int = None):
        model.train()

        # print(type(inputs))
        #
        # # is this needed?
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.args.device)
        # TODO: why is the loss just the output?

        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        params = None
        loss = 0
        for i in range(int(self.num_inner_steps)):
            # TODO: what are inputs here?
            outputs = model(**inputs, params=params)
            step_loss = outputs[0].mean()
            loss += step_loss
            params = gradient_update_parameters(model,
                                                params=params,
                                                loss=step_loss,
                                                # TODO: This is likely not right - doesn't decay, but could be fine
                                                step_size=self.args.learning_rate,
                                                first_order=self.first_order)

        # if self.args.n_gpu > 1:
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # if self.args.gradient_accumulation_steps > 1:
        #     loss = loss / self.args.gradient_accumulation_steps
        #
        # if self.args.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()

        if is_mlflow_available() and outer_step:
            # personally, I only want this logged every 10 steps or so
            if outer_step % 10 == 0:
                mlflow.log_metric('training/inner_loss', loss.item(), outer_step)

        return loss.item(), params

    def _outer_training_step(self, model: nn.Module, params: OrderedDict, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)
        # TODO: why is the loss just the output?
        model.zero_grad()
        outputs = model(**inputs, params=params)
        loss = outputs[0].mean()
        loss.backward()

        return loss.item()
