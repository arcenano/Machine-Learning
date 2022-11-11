import argparse  # Interface
import json  # Load Config file
from loguru import logger  # Simplified Logging
import pathlib  # Path verification
import datetime  # Get date for name
import wandb  # Save weights on the cloud
from pytorch_lightning.trainer import Trainer  # Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback  # Checkpoints
from pytorch_lightning.loggers import WandbLogger  # Logging
from easydict import EasyDict  # Parsing Config File
from FOTS.model.model import FOTSModel  # Model Lightning Module
from FOTS.dataloader.ICDARDataModule import ICDARDataModule  # Dataloaders
from FOTS.dataloader.SynthTextDataModule import SynthTextDataModule  # Dataloaders
import atexit  # Running funtions after program execution (deleting files)
import os  # Create path object for deleting with shutil
import shutil  # Deleting files
from wb.load_wandb import InitWandb, LoadWeights # WandB init and loading model
from FOTS.utils.bbox import Toolbox
import torch
import traceback


def main(config):

    # Get weights file
    model_path = LoadWeights(config)

    # Set input and output directories
    input_dir = pathlib.Path(config.input_dir)
    output_dir = pathlib.Path(config.output_dir)
    
    # Make sure input directory exists
    assert input_dir.exists()

    # Create output directory if it doesn't exist
    if output_dir:
        if not output_dir.exists():
            print("making checkpoints path")
            output_dir.mkdir(parents=True, exist_ok=True)

    # Save Images
    with_image = True if output_dir else False

    # GPU Availability
    with_gpu = False #True if torch.cuda.is_available() else False

    # Load model with weights
    model = FOTSModel.load_from_checkpoint(
        checkpoint_path=model_path, map_location="cpu", config=config
    )
    
    # Turn off batchnorm & dropout
    model.eval()

    print("Ready to test")

    for image_fn in input_dir.glob("*.jpg"):
        try:
            # Turn off gradient calculations
            with torch.no_grad():

                # Get predictions
                ploy, im = Toolbox.predict(
                    image_fn, model, with_image, output_dir, with_gpu=with_gpu
                )
                # Print model parameters
                # print(model.state_dict())
                print(len(ploy))

        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args = parser.parse_args()

    config = json.load(open(args.config))

    config = EasyDict(config)
    print(f"Successfully read config file:\n{config}")

    main(config)