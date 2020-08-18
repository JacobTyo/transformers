import mlflow
import logging
import torch


class Logger(logging.Logger):
    def __init__(self, name, args, level=logging.info):
        super(Logger, self).__init__(name)
        # TODO: setup MLFlow stuff
        try:
            for key, val in vars(args).items():
                mlflow.log_param(key, val)
        except Exception as e:
            for key, val in args.items():
                mlflow.log_param(key, val)
        self.name = name
        self.args = args
        self.best_model_loss = None

    @staticmethod
    def log_train(step, loss, outer_loss=None):
        mlflow.log_metric('training/loss', loss, step)
        if outer_loss is not None:
            mlflow.log_metric('training/outer_loss', outer_loss, step)

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

    def save_model(self, model, save_path, loss):
        if self.best_model_loss is None or self.best_model_loss >= loss:
            self.info('saving best model')
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
        else:
            self.info('model performance is not best, not saving')