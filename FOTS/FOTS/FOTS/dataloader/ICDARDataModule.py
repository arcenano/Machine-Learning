from typing import Optional, Any, Union, List # Defining Types
from easydict import EasyDict # Config file type
from pytorch_lightning import LightningDataModule # Module inheritance
from torch.utils.data import DataLoader # Pytorch Dataloader class
from .ICDARDataSet import ICDARDataset # Dataset
from .Transform import Transform # Class for image transforming and augmentation 
from .datautils import collate_fn # Batch loading logic 

class ICDARDataModule(LightningDataModule):
    def __init__(self, config: EasyDict):
        super(ICDARDataModule, self).__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        transform = Transform(
            is_training=False,
            output_size=(self.config.data_loader.size, self.config.data_loader.size),
        )
        self.train_ds = ICDARDataset(
            data_root=self.config.data_loader.data_dir + "/train",
            transform=transform,
            vis=False,
            training=True,
            size=self.config.data_loader.size,
            scale=self.config.data_loader.scale,
        )

        transform = Transform(is_training=False)
        self.val_ds = ICDARDataset(
            data_root=self.config.data_loader.data_dir + "/test",
            transform=transform,
            vis=False,
            training=False,
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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.config.data_loader.batch_size,
            num_workers=self.config.data_loader.workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=False,
        )
