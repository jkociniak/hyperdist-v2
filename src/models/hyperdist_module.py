from typing import Any, List

import torch.nn as nn
import torch
import geoopt
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanAbsolutePercentageError, ExplainedVariance, R2Score
from components.mixed_model import StandardMixedModel, TrueModel, TrueHeadModel, TrueEmbeddingModel
from components.euclidean_mlp import DoubleInputEuclideanMLP


class HYPERDISTLitModule(LightningModule):
    name2model = {
        'EuclideanModel': DoubleInputEuclideanMLP,
        'StandardMixedModel': StandardMixedModel,
        'TrueModel': TrueModel,
        'TrueEmbeddingModel': TrueEmbeddingModel,
        'TrueHeadModel': TrueHeadModel,
    }

    name2optimizer = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }

    name2roptimizer = {
        'RiemannianSGD': geoopt.optim.RiemannianSGD,
        'RiemannianAdam': geoopt.optim.RiemannianAdam
    }

    name2scheduler = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    def __init__(self,
                 net: nn.Module,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 scheduler_name: str,
                 scheduler_hparams: dict,
                 roptimizer_name: str,
                 roptimizer_hparams: dict,
                 rscheduler_name: str,
                 rscheduler_hparams: dict):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.build_model(self.hparams.model_name, **self.hparams.model_hparams)
        # Create loss module
        self.loss_module = torch.nn.MSELoss()

        self.metrics = {}
        for set in ['train', 'val', 'test']:
            self.metrics[set] = {
                'MAPE': MeanAbsolutePercentageError(),
                'ExplainedVariance': ExplainedVariance(),
                'R2': R2Score()
            }

        self.val_metrics_best = {
            'MAPE': MaxMetric(),
            'ExplainedVariance': MaxMetric(),
            'R2': MaxMetric()
        }

        # Example input for visualizing the graph in Tensorboard
        #self.example_input_array = torch.zeros((1, input_dim, input_dim), dtype=torch.float32)

    def forward(self, pairs: geoopt.ManifoldTensor):
        # Forward function that is run when visualizing the graph
        return self.model(pairs)

    def step(self, batch: Any):
        # "batch" is the output of the training data loader.
        pairs, dists = batch['pairs'], batch['dist']
        x1, x2 = pairs[:, 0, :], pairs[:, 1, :]

        preds = self.model(x1, x2)
        loss = self.loss_module(preds, dists)
        return loss, preds, dists

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, dists = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # log metrics
        for name, metric in self.metrics['train'].items():
            m = metric(preds, dists)
            self.log(f'train/{name}', m, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': loss, 'preds': preds, 'dists': dists}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, dists = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # log metrics
        for name, metric in self.metrics['val'].items():
            m = metric(preds, dists)
            self.log(f'train/{name}', m, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': loss, 'preds': preds, 'dists': dists}

    def validation_epoch_end(self, outputs: List[Any]):
        for name, max_metric in self.val_metrics_best:
            m = self.metrics['val'][name].compute()  # get val metric from current epoch
            max_metric.update(m)
            self.log(f'val/{name}_best', max_metric.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, dists = self.step(batch)
        self.log('train/loss', loss, on_step=False, on_epoch=True)

        # log metrics
        for name, metric in self.metrics['test'].items():
            m = metric(preds, dists)
            self.log(f'train/{name}', m, on_step=False, on_epoch=True)

        return {'loss': loss, 'preds': preds, 'dists': dists}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        for metrics_dict in self.metrics.values():
            for metric in metrics_dict.values():
                metric.reset()

    def build_model(self, name, **kwargs):
        """
        Function used to build model based on its name.
        :param name: name of the model
        :param kwargs: parameters to pass to the model constructor
        :return: object containing built model
        """
        try:
            return self.name2model[name](**kwargs)
        except KeyError:
            raise NotImplementedError(f'model {name} is not implemented!')

    def build_optimizer(self, name, params, **kwargs):
        """
        Function used to build optimizer based on its name.
        :param name: name of the optimizer
        :param params: parameters to be optimized
        :param kwargs: parameters to pass to the optimizer constructor
        :return: object containing built optimizer
        """
        try:
            return self.name2optimizer[name](params, **kwargs)
        except KeyError:
            raise NotImplementedError(f'optimizer {name} is not implemented!')

    def build_roptimizer(self, name, params, **kwargs):
        """
        Function used to build Riemannian optimizer based on its name.
        :param name: name of the optimizer
        :param params: parameters to be optimized
        :param kwargs: parameters to pass to the optimizer constructor
        :return: object containing built optimizer
        """
        try:
            return self.name2roptimizer[name](params, **kwargs)
        except KeyError:
            raise NotImplementedError(f'Riemannian optimizer {name} is not implemented!')

    def build_scheduler(self, name, opt, **kwargs):
        """
        Function used to build scheduler based on its name.
        :param name: name of the scheduler
        :param opt: optimizer to be scheduler
        :param kwargs: parameters to pass to the scheduler constructor
        :return: object containing built scheduler
        """
        try:
            return self.name2scheduler[name](opt, **kwargs)
        except KeyError:
            raise NotImplementedError(f'scheduler {name} is not implemented!')

    def configure_optimizers(self):
        opt = self.build_optimizer(self.parameters(), self.hparams.optimizer_name, **self.hparams.optimizer_hparams)
        ropt = self.build_roptimizer(self.parameters(), self.hparams.roptimizer_name, **self.hparams.roptimizer_hparams)

        sch = self.build_scheduler(opt, self.hparams.scheduler_name, **self.hparams.scheduler_hparams)
        rsch = self.build_scheduler(ropt, self.hparams.rscheduler_name, **self.hparams.rscheduler_hparams)
        return [opt, ropt], [sch, rsch]
