import torch
import os
import datetime
import copy
from torch.utils.tensorboard import SummaryWriter
from bispectral_networks.config import Config

        
class TBLogger:
    def __init__(
        self,
        config,
        log_interval=1,
        checkpoint_interval=10,
        logdir=None
    ):
        self.config = config
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.logdir = logdir

    def begin(self, model, data_loader):
        try:
            self.create_logdir()
            torch.save(self.config, os.path.join(self.logdir, "config.pt"))
            writer = SummaryWriter(self.logdir)
            return writer
        except:
            raise Exception("Problem creating logging and/or checkpoint directory.")
            
    def end(self, trainer, variable_dict, epoch):
        self.save_checkpoint(trainer, epoch)

    def create_logdir(self):
        if self.logdir is None:
            self.logdir = os.path.join(
                "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )

        os.makedirs(self.logdir, exist_ok=True)
        os.mkdir(os.path.join(self.logdir, "checkpoints"))
            
    def log_step(self, writer, trainer, log_dict, variable_dict, epoch, val_log_dict=None):
        if epoch % self.log_interval == 0:
            writer.add_scalars("train", log_dict, global_step=epoch)
            if val_log_dict is not None:
                writer.add_scalars("val", val_log_dict, global_step=epoch)

        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(trainer, epoch)
                                
    def save_checkpoint(self, trainer, epoch):
        checkpoint = {
            "trainer": trainer,
            "model_state_dict": trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(self.logdir, "checkpoints", "checkpoint_{}.pt".format(epoch)),
        )

        
def load_checkpoint(logdir, device="cpu"):
    all_checkpoints = os.listdir(os.path.join(logdir, "checkpoints"))
    all_epochs = sorted([int(x.split("_")[1].split(".")[0]) for x in all_checkpoints])
    last_epoch = all_epochs[-1]
    checkpoint = torch.load(os.path.join(logdir, “checkpoints”, “checkpoint_{}.pt”.format(last_epoch)), map_location=torch.device(device))
    config = torch.load(os.path.join(logdir, "config.pt"))
    if not hasattr(checkpoint, "model"):
        trainer = checkpoint["trainer"]
        model_config = Config(trainer.logger.config["model"])
        optimizer_config = Config(copy.deepcopy(trainer.logger.config["optimizer"]))
        trainer.model = model_config.build()
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_config["params"]["params"] = trainer.model.parameters()
        trainer.optimizer = optimizer_config.build()
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = trainer
    W = (checkpoint.model.layers[0].W.real + 1j * checkpoint.model.layers[0].W.imag).detach().numpy()
    patch_size = config["dataset"]["pattern"]["params"]["patch_size"]
    W = W.reshape(W.shape[0], patch_size, patch_size)
    return checkpoint, config, W
