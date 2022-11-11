import pathlib  # Path verification
import datetime  # Get date for name
import wandb  #Save weights on the cloud
from pytorch_lightning.callbacks import ModelCheckpoint, Callback  # Checkpoints

# WandB cloud upload of checkpoint
class WandBsave(Callback):
    def __init__(self, checkpoints_path):
        self.dir = str(checkpoints_path)
        self.maindir = wandb.run.dir

    def on_train_batch_start(
        self, lightning, outputs, batch, batch_idx, dataloader_idx
    ): 
        wandb.save(self.dir + "/" + "*.ckpt", self.maindir, "now")
        ## Delete file from previous run

    # def on_keyboard_interrupt(self, lightning, outputs, batch, batch_idx, dataloader_idx):
    #     wandb.save(self.dir + "/" + "*.ckpt", self.maindir, "now")

    # save on key interrupt & on exception

    # def on_exception(self, lightning, outputs, batch, batch_idx, dataloader_idx):

# WandB cloud upload of checkpoint
class Checkpoints():
    def __init__(self, save_interval):
        self.checkpoints_path = self.MakeCheckpointsPath()

        # Convert save interval minutes into timedelta object
        self.save_interval = datetime.timedelta(minutes=save_interval)

    def ModelCP(self):
        # Model weights saving checkpoint callback
        return ModelCheckpoint(
            dirpath=self.checkpoints_path, train_time_interval=self.save_interval
        )  # every_n_train_steps/every_n_epochs

    def WBsave(self):
        return WandBsave(self.checkpoints_path)

    def MakeCheckpointsPath(self):
        checkpoints_path = pathlib.Path(wandb.run.dir) / "checkpoints"

        print("====CKPS PTH====\n\n")
        print(checkpoints_path)

        if not checkpoints_path.exists():
            print("making checkpoints path")
            checkpoints_path.mkdir(parents=True, exist_ok=True)
        
        return checkpoints_path