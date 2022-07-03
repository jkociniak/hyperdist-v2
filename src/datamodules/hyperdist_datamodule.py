from typing import Optional

import os
import pickle
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset
from components.HyperdistDataset import HYPERDIST


class HYPERDISTDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        seed: int = 777,
        n_samples: int = 100000,
        dim: int = 2,
        curv: float = 1,
        inverse_transform: str = 'hyperbolic',
        min_r: float = 0.1,
        max_r: float = 5.3,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.dataset_name = self.build_dataset_path(**self.hparams)
        self.dataset_path = os.path.join(self.hparams.data_dir, self.dataset_name)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        if not os.path.isfile(self.dataset_path):
            self.build_datasets(**self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            with open(self.dataset_path, 'rb') as f:
                datasets = pickle.load(f)
                loaders = self.build_loaders(datasets, self.hparams.batch_size, self.hparams.num_workers, self.hparams.pin_memory)
                self.data_train, self.data_val, self.data_test = loaders['train'], loaders['val'], loaders['test']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def build_dataset_path(self, seed, n_samples, dim, curv, inverse_transform, min_r, max_r):
        template = 'd={},ns={},c={},seed={},it={},minr={},maxr={}.pkl'
        return template.format(int(dim), int(n_samples), curv, seed, inverse_transform, min_r, max_r)

    def build_datasets(self, seed, n_samples, dim, curv, inverse_transform, min_r, max_r, data_dir):
        seed_everything(seed)  # reset RNGs before dataset generation

        n_train, n_val = int(0.7 * n_samples), int(0.2 * n_samples)

        sizes = {
            'train': n_train,
            'val': n_val,
            'test': int(n_samples - n_train - n_val)
        }

        print('Generating datasets...')
        datasets = {}
        for name, size in sizes.items():
            print(f'Processing {name} set of size {size}...')
            dataset = HYPERDIST(size, dim, curv, inverse_transform, min_r, max_r)
            datasets[name] = dataset

        filename = self.build_dataset_path(seed, n_samples, dim, curv, inverse_transform, min_r, max_r)
        filepath = os.path.join(data_dir, filename)
        print(f'Saving datasets at path {filepath}')
        with open(filepath, 'wb') as f:
            pickle.dump(datasets, f)

    def build_loaders(self, datasets, bs, num_workers, pin_memory):
        loaders = {name: self.build_dataloader(name, dataset, bs, num_workers, pin_memory)
                   for name, dataset in datasets.items()}
        return loaders

    def build_dataloader(self, name, dataset, bs, num_workers, pin_memory):
        if name == 'train':
            return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        else:
            return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
