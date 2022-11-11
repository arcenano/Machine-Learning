from typing import Optional, Any # Defining types
from easydict import EasyDict # Confing type definition
from pytorch_lightning import LightningDataModule # Module inheritance
from torch.utils.data import DataLoader # Pytorch Dataloader class
from .SynthTextDataSet import SynthTextDataset # Dataset
from .Transform import Transform # Class for image transforming and augmentation 
from .datautils import collate_fn # Batch loading logic 

class SynthTextDataModule(LightningDataModule):
    def __init__(self, config: EasyDict):
        super(SynthTextDataModule, self).__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        traintransform = Transform(
            is_training=True,
            output_size=(self.config.data_loader.size, self.config.data_loader.size),
        )

        testtransform = Transform(
            is_training=False,
            output_size=(self.config.data_loader.size, self.config.data_loader.size),
        )
        
        self.train_ds = SynthTextDataset(
            data_root=self.config.data_loader.data_dir,
            transform=traintransform,
            vis=False,
            size=self.config.data_loader.size,
            scale=self.config.data_loader.scale,
        )

        self.test_ds = SynthTextDataset(
            data_root=self.config.data_loader.data_dir,
            transform=testtransform,
            vis=False,
            size=self.config.data_loader.size,
            scale=self.config.data_loader.scale,
        )

    def train_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=self.config.data_loader.shuffle,
            pin_memory=False,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=self.config.data_loader.shuffle,
            pin_memory=False,
        )
