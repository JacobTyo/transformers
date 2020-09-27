import mlflow
import logging
import torch
import json
import threading

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
    def log_eval(avg_loss, avg_perplexity, step, loss_std=None, perplex_std=None, loss_ste=None, perplex_ste=None):
        mlflow.log_metric('validation/loss_avg', avg_loss, step)
        mlflow.log_metric('validation/perplexity_avg', avg_perplexity, step)
        if loss_std:
            mlflow.log_metric('validation/loss_std', loss_std, step)
        if perplex_std:
            mlflow.log_metric('validation/perplexity_std', perplex_std, step)
        if loss_ste:
            mlflow.log_metric('validation/loss_ste', loss_ste, step)
        if perplex_ste:
            mlflow.log_metric('validation/perplexity_ste', perplex_ste, step)

    def log_json(self, json_obj, filename='temporary_json_file.json'):
        # save the object to a temporary place
        with open(filename, 'w') as f:
            json.dump(json_obj, f, indent=4, sort_keys=True)
        mlflow.log_artifact(filename)

    @staticmethod
    def log_test(loss, perplexity, step, std=None):
        mlflow.log_metric('test/loss', loss, step)
        mlflow.log_metric('test/perplexity', perplexity, step)
        if std:
            mlflow.log_metric('test/std', std, step)

    # TODO: Make sure this works, I'm currently losing my models I think
    def save_model(self, model, optimizers, save_path, loss, step):
        savethread = threading.Thread(target=self._save_model,
                                      args=(model,
                                            optimizers,
                                            save_path,
                                            loss,
                                            step))
        # start the thread
        savethread.daemon = True
        savethread.start()

    def _save_model(self, model, optimizers, save_path, loss, step):
        # if self.best_model_loss is None or self.best_model_loss >= loss:
        #     self.info('saving best model')
        try:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizers[0].state_dict(),
                'schedular_state_dict': optimizers[1].state_dict(),
                'loss': loss
            }, save_path)
        except Exception:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizers.state_dict(),
                'schedular_state_dict': None,
                'loss': loss
            }, save_path)

        mlflow.log_artifact(save_path)
        # else:
        #     self.info('model performance is not best, not saving')