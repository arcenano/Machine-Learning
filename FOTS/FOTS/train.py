import argparse  # Interface
import json  # Load Config file
from easydict import EasyDict  # Parsing Config File
from pytorch_lightning.trainer import Trainer  # Trainer
from FOTS.model.model import FOTSModel  # Model Lightning Module
from FOTS.dataloader.DataModule import DataModule  # Dataloaders
from wb.load_wandb import InitWandb, LoadWeights # WandB init and loading model
from wb.checkpoint import Checkpoints # Model Checkpoint Save & Upload
import torch

def main(config):

    # Empty the cuda memory that we might have used before
    torch.cuda.empty_cache()

    # Load Model
    model = FOTSModel(config)

    # Load Data Module
    data_module = DataModule(config)

    # Initialize WandB
    wandb_logger = InitWandb(config)

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")    

    # Resume CKPT
    resume_ckpt = LoadWeights(config)

    # Class thats given the save interval and the cp path and its
    checkpoints = Checkpoints(config.save_interval)

    # Model weights saving checkpoint callback
    checkpoint_callback = checkpoints.ModelCP()

    # WandB callback to upload model weights
    wandb_callback = checkpoints.WBsave()

    # Verify gpu use
    if not config.cuda.on:
        gpus = 0
    else:
        gpus = [0]

    # Configure stradegy
    if not config.cuda.strategy:
        strategy = "dp"
        num_nodes = 1
    # accelerator = None
    else:
        strategy = "dpp"
        # accelerator = "gpu"
        num_nodes = config.cuda.num_nodes

    trainer = Trainer(
        logger=wandb_logger,  # Trainer logging
        callbacks=[checkpoint_callback, wandb_callback],  # Checkpoint path
        max_epochs=config.trainer.epochs, # Training Epochs
        #auto_lr_find=True, # Find optimal LR
        gpus=gpus, # GPUs in use
        # Distributed training
        # num_nodes=num_nodes,  # If distributed, number of nodes
        # accelerator=accelerator,  # Type of processor being used
        benchmark=True,  # Speeds up trainer if all images are the same input size
        sync_batchnorm=True,  # Changes batch normalization so its taken out of the entire sample and not only on local gpu's data
        precision=config.precision,  # Changes the precision on floating point numbers (16 bit, 32 bit or 64 bit)
        log_gpu_memory=config.trainer.log_gpu_memory,  # Logs the memory usage per GPU
        log_every_n_steps=config.trainer.log_every_n_steps,  # Log every N batches
        overfit_batches=config.trainer.overfit_batches,  # Train the model to overfit to test logic bugs
        weights_summary="top",  # Prints a summary of the model architecture (layers)
        terminate_on_nan=config.trainer.terminate_on_nan,  # If set to True, will terminate training at the end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
        fast_dev_run=config.trainer.fast_dev_run,  # Runs every line of code in the model to check for bugs
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,  # Runs the validation code every N epochs, default is 1
        resume_from_checkpoint=resume_ckpt,  # Resume training from a saved checkpoint with path
    )

    print("Ready to Train")

    trainer.fit(model=model, datamodule=data_module)

    

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
