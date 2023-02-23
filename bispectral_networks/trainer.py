import numpy as np
import torch
from collections import OrderedDict
import copy
from bispectral_networks.config import Config
from bispectral_networks.data.utils import gen_dataset


class BispectralTrainer(torch.nn.Module):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        recon_coeff=150,
        logger=None,
        scheduler=None,
        normalizer=None,
    ):
        super().__init__()
        self.recon_coeff = recon_coeff
        self.model = model
        self.loss = loss
        self.logger = logger
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.n_examples = 0
        
    def __getstate__(self):
        d = self.__dict__
        self_dict = {k : d[k] for k in d if k != '_modules'}
        module_dict = OrderedDict({'loss': self.loss})
        if self.normalizer is not None:
            module_dict["normalizer"] = self.normalizer
        if self.scheduler is not None:
            module_dict["scheduler"] = self.scheduler
        self_dict['_modules'] = module_dict
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state

    def step(self, data_loader, grad=True):
        log_dict = {"loss": 0, "rep_loss": 0, "recon_loss": 0, "total_loss": 0}
        for i, (x, labels) in enumerate(data_loader):
            loss = 0
            total_loss = 0

            x = x.to(self.model.device)
            labels = labels.to(self.model.device)

            if grad:
                self.optimizer.zero_grad()
                out, recon = self.model.forward(x)

            else:
                with torch.no_grad():
                    out, recon = self.model.forward(x)

            rep_loss = abs(self.loss(out, labels))
            recon_loss = self.recon_coeff * torch.nn.functional.mse_loss(recon, x.float())
            
            log_dict["rep_loss"] += rep_loss
            log_dict["recon_loss"] += recon_loss
            total_loss += rep_loss + recon_loss

            if grad:
                total_loss.backward()
                self.optimizer.step()
                
            if self.normalizer is not None:
                self.normalizer(dict(self.model.named_parameters()))

            log_dict["total_loss"] += total_loss

        n_samples = len(data_loader)
        for key in log_dict.keys():
            log_dict[key] /= n_samples

        plot_variable_dict = {"model": self.model}

        return log_dict, plot_variable_dict
    
    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        print_interval=1,
    ):
        if self.logger is not None:
            writer = self.logger.begin(self.model, data_loader)

        try:
            for i in range(start_epoch, start_epoch + epochs + 1):
                self.epoch = i
                log_dict, plot_variable_dict = self.step(data_loader.train, grad=True)
                

                if data_loader.val is not None:
                    # By default, plots are only generated on train steps
                    val_log_dict, _ = self.evaluate(
                        data_loader.val
                    ) 
                else:
                    val_log_dict = None
                    
                if self.scheduler is not None:
                    if val_log_dict is not None:
                        self.scheduler.step(val_log_dict["total_loss"])
                    else:
                        self.scheduler.step(train_log_dict["total_loss"])

                if self.logger is not None:
                    self.logger.log_step(
                        writer=writer,
                        trainer=self,
                        log_dict=log_dict,
                        val_log_dict=val_log_dict,
                        variable_dict=plot_variable_dict,
                        epoch=self.epoch,
                    )
                    
                self.n_examples += len(data_loader.train.dataset)

                if i % print_interval == 0 and print_status_updates == True:
                    if data_loader.val is not None:
                        self.print_update(log_dict, val_log_dict)
                    else:
                        self.print_update(log_dict)


        except KeyboardInterrupt:
            print("Stopping and saving run at epoch {}".format(i))
        end_dict = {"model": self.model, "data_loader": data_loader}
        if self.logger is not None:
            self.logger.end(self, end_dict, self.epoch)

    def resume(self, data_loader, epochs):
        self.train(data_loader, epochs, start_epoch=self.epoch+1)

    @torch.no_grad()
    def evaluate(self, data_loader):
        results = self.step(data_loader, grad=False)
        return results

    def print_update(self, result_dict_train, result_dict_val=None):

        update_string = "Epoch {} ||  N Examples {} || Train Total Loss {:0.5f}".format(
            self.epoch, self.n_examples, result_dict_train["total_loss"]
        )
        if result_dict_val:
            update_string += " || Validation Total Loss {:0.5f}".format(
                result_dict_val["total_loss"]
            )
        print(update_string)


def construct_trainer(master_config, logger_config=None):
    """
    master_config has the following format:
    master_config = {
        "dataset": dataset_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "loss": loss_config,
        "data_loader": data_loader_config,
    }
    with optional regularizer, normalizer, and learning rate scheduler
    """
    
    if "seed" in master_config:
        torch.manual_seed(master_config["seed"])
        np.random.seed(master_config["seed"])
        
    model = master_config["model"].build()
    loss = master_config["loss"].build()
    
    logger_config["params"]["config"] = master_config
    logger = logger_config.build()

    optimizer_config = copy.deepcopy(master_config["optimizer"])
    optimizer_config["params"]["params"] = model.parameters()
    optimizer = optimizer_config.build()
        
    train_config = Config(
        {
            "type": BispectralTrainer,
            "params": {
                "model": model,
                "loss": loss,
                "logger": logger,
                "optimizer": optimizer,
            },
        }
    )

    if "regularizer" in master_config:
        regularizer = master_config["regularizer"].build()
        train_config["params"]["regularizer"] = regularizer

    if "normalizer" in master_config:
        normalizer = master_config["normalizer"].build()
        train_config["params"]["normalizer"] = normalizer
        
    if "scheduler" in master_config:
        scheduler_config = copy.deepcopy(master_config["scheduler"])
        scheduler_config["params"]["optimizer"] = optimizer
        scheduler = scheduler_config.build()
        train_config["params"]["scheduler"] = scheduler
        
    trainer = train_config.build()
    
    return trainer


def run_trainer(master_config, 
                logger_config,
                device=0, 
                n_examples=1e9,
                seed=None):

    if seed is not None:
        master_config["seed"] = seed
        
    dataset = gen_dataset(master_config["dataset"])

    data_loader = master_config["data_loader"].build()
    data_loader.load(dataset)

    trainer = construct_trainer(master_config, logger_config=logger_config)

    epochs = int(n_examples // len(data_loader.train.dataset.data))
    trainer.model.device = device
    trainer.model = trainer.model.to(device)
    trainer.train(data_loader, epochs=epochs)
