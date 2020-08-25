import mlflow
import logging
import torch


class Logger(logging.Logger):
    def __init__(self, name, args=None, level=logging.info, artifact_uri=None):
        super(Logger, self).__init__(name)
        # TODO: setup MLFlow stuff
        if args is not None:
            try:
                for key, val in vars(args).items():
                    mlflow.log_param(key, val)
            except Exception as e:
                for key, val in args.items():
                    mlflow.log_param(key, val)
        self.name = name
        self.args = args
        self.best_model_loss = None
        self.artifact_uri = artifact_uri

    @staticmethod
    def log_train(step, loss, outer_loss=None):
        mlflow.log_metric('training/loss', loss, step)

    @staticmethod
    def log_outer(step, loss):
        mlflow.log_metric('training/outer_loss', loss, step)

    # TODO: review from here below
    @staticmethod
    def log_eval(loss, perplexity, step, std=None):
        mlflow.log_metric('validation/loss', loss, step)
        mlflow.log_metric('validation/perplexity', perplexity, step)
        if std:
            mlflow.log_metric('validation/std', std, step)

    @staticmethod
    def log_test(loss, perplexity, step, std=None):
        mlflow.log_metric('test/loss', loss, step)
        mlflow.log_metric('test/perplexity', perplexity, step)
        if std:
            mlflow.log_metric('test/std', std, step)

    # TODO: Make sure this works, I'm currently losing my models I think
    def save_model(self, model, optimizer, save_path, loss, step):
        if self.best_model_loss is None or self.best_model_loss >= loss:
            self.info('saving best model')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)

            mlflow.log_artifact(save_path, artifact_path=self.artifiact_uri)
        else:
            self.info('model performance is not best, not saving')
