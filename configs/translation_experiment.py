from bispectral_networks.config import Config

"""
DATA_LOADER
"""
from bispectral_networks.data.data_loader import MPerClassLoader

data_loader_config = Config(
    {
        "type": MPerClassLoader,
        "params": {
            "batch_size": 100,
            "m": 10,
            "fraction_val": 0.2,
            "num_workers": 1,
        },
    }
)

"""
DATASET
"""
from bispectral_networks.data.transforms import CyclicTranslation2D, Ravel, CenterMean, UnitStd
from bispectral_networks.data.datasets import VanHateren


pattern_config = Config(
    {
        "type": VanHateren,
        "params": {"path": "datasets/van-hateren/",
                   "min_contrast": 0.1,
                   "patches_per_image": 3,
                   "patch_size": 16},
    }
)

transforms_config = {
    "0": Config(
        {
            "type": CenterMean,
            "params": {}
        }
    ),
    "1": Config(
        {
            "type": UnitStd,
            "params": {}
        }
    ),
    "2": Config(
        {
            "type": CyclicTranslation2D,
            "params": {
                "fraction_transforms": 1.0,
                "sample_method": "random"
            },
        }
    ),
    "3": Config(
        {
            "type": Ravel,
            "params": {},
        }
    )
}


dataset_config = {"pattern": pattern_config, 
                  "transforms": transforms_config,
                  "seed": 5}


"""
MODEL
"""
from bispectral_networks.nn.model import BispectralEmbedding
from bispectral_networks.nn.layers import Bispectral

model_config = Config(
    {
        "type": BispectralEmbedding,
        "params": {"size_in": 256, 
                   "hdim": 256},
    }
)


"""
NORMALIZER
"""
from bispectral_networks.normalizer import L2Normalizer

normalizer_config = Config({
    "type": L2Normalizer,
    "params": {
        "variables": ["layers.0.W"]
    }
    
})


"""
LOSS
"""
from pytorch_metric_learning import losses, reducers
from bispectral_networks.loss import OrbitCollapse, LpDistance
loss_config = Config(
    {
        "type": OrbitCollapse,
        "params": {
            "distance": LpDistance(),
        },
    }
)


"""
OPTIMIZER
"""
from torch.optim import Adam

optimizer_config = Config({"type": Adam, "params": {"lr": 0.002}})


"""
SCHEDULER
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler_config = Config({"type": ReduceLROnPlateau, "params": {"factor": 0.5, "patience": 2, "min_lr": 1e-6}})


"""
LOGGER
"""
   
from bispectral_networks.logger import TBLogger

logger_config = Config(
    {
        "type": TBLogger,
        "params": {
            "log_interval": 1,
            "checkpoint_interval": 10,
        },
    }
)


"""
MASTER CONFIG
"""

master_config = {
    "data_loader": data_loader_config,
    "dataset": dataset_config,
    "model": model_config,
    "optimizer": optimizer_config,
    "normalizer": normalizer_config,
    "scheduler": scheduler_config,
    "loss": loss_config,
    "seed": 200
}