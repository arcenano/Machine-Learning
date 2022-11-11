from typing import Optional, Any  # Defining types
from easydict import EasyDict  # Confing type definition
from pytorch_lightning import LightningDataModule  # Module inheritance
from .ICDARDataSet import ICDARDataset  # Dataset
from .SynthTextDataSet import SynthTextDataset  # Dataset
from pytorch_lightning import LightningDataModule  # Module inheritance
from torch.utils.data import DataLoader  # Pytorch Dataloader class
from .Transform import Transform  # Class for image transforming and augmentation
from .datautils import collate_fn  # Batch loading logic


class DataModule(LightningDataModule):
    def __init__(self, config: EasyDict):
        super(DataModule, self).__init__()
        self.config = config
        print("Loading Data Module")

    def setup(self, stage: Optional[str] = None):
        print("Setting up DataModule")
        train = self.config.data_loader.train
        test = self.config.data_loader.test
        val = self.config.data_loader.val

        self.train_ds = self.findDataSet(train, True)
        self.test_ds = self.findDataSet(test, False)
        self.validation_ds = self.findDataSet(val, False)

        print("DataModule has finished setting up")

    def findDataSet(self, info, isTrain):
        if isTrain:
            transform = Transform(
                is_training=True,
                output_size=(
                    self.config.data_loader.size,
                    self.config.data_loader.size,
                ),
            )
        else:
            transform = Transform(
                is_training=False,
                output_size=(
                    self.config.data_loader.size,
                    self.config.data_loader.size,
                ),
            )

        text = info.dataset

        if text == "synthtext":
            return SynthTextDataset(
                data_root=info.dir,
                transform=transform,
                vis=False,
                size=self.config.data_loader.size,
                scale=self.config.data_loader.scale,
            )
        elif text == "icdar2015":
            return ICDARDataset(
                data_root=info.dir,
                transform=transform,
                vis=False,
                size=self.config.data_loader.size,
                scale=self.config.data_loader.scale,
            )
        else:
            return KeyError("Dataset not found")

    def initialize(self):
        train = self.config.data_loader.train
        test = self.config.data_loader.test
        val = self.config.data_loader.val

        train_dataset = self.findDataSet(train, True)
        test_dataset = self.findDataSet(test, False)
        val_dataset = self.findDataSet(val, False)

        return train_dataset, test_dataset, val_dataset

    def train_dataloader(self) -> Any:
        print("Loading Training DataLoader")
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=self.config.data_loader.shuffle,
            pin_memory=False,
        )

    def val_dataloader(self) -> Any:
        print("Loading Validation DataLoader")
        return DataLoader(
            dataset=self.validation_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> Any:
        print("Loading Test DataLoader")
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=False,
        )