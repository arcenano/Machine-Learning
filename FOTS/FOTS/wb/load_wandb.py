from loguru import logger  # Simplified Logging
import pathlib  # Path verification
import datetime  # Get date for name
from pytorch_lightning.loggers import WandbLogger  # Logging
import wandb  # Save weights on the cloud

def InitWandb(config):
    # Append date to name
    date = datetime.datetime.now()
    name = str(config.name) + date.strftime(" @ %Y-%m-%d %H:%M")

    # Run ID (for resuming)
    if not config.wandb_id:
        id = wandb.util.generate_id()
        print("Starting new run with id:" + id)

    else:
        id = config.wandb_id
        print("Continuing run with id:" + id)

    # Initialize WandB
    wandb.init(name=name, project="FOTS", config=config, id=id, resume="allow")
    # Instantiate WandB Logger
    wandb_logger = WandbLogger(name=name, project="FOTS", config=config)

    return wandb_logger

def LoadWeights(config):
    # Resuming Training

    if config.wandb_run:
        # If wandb run path is given load ckpt from file specified in pretrain
        print("Reading weights from" + config.wandb_run + "/" + config.pretrain)
        logger.info("Reading weights from " + config.wandb_run + "/" + config.pretrain)

        # Get weights file from wandb
        weights_file = wandb.restore(
            config.pretrain, run_path=config.wandb_run
        )  # wandb_run ex: "arcenano/FOTS/runs/3b8xd8ef"
        resume_ckpt = weights_file.name

    elif config.pretrain:
        # If ckpt file must be read
        if config.wandb_id:
            # If wandb run is resumed load checkpoint from ID save directory
            print("Reading weights from resumed run at " + config.pretrain)
            logger.info("Reading weights from resumed run at " + config.pretrain)

            # Get weights file from wandb
            weights_file = wandb.restore(config.pretrain)
            resume_ckpt = weights_file.name

        else:
            # Load file from local path
            assert pathlib.Path(config.pretrain).exists()
            resume_ckpt = config.pretrain
            print("Resume training from local file: " + config.pretrain)
            logger.info("Resume training from local file:" + config.pretrain)

    else:
        resume_ckpt = None

    return resume_ckpt