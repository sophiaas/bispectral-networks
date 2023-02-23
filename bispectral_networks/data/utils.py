from bispectral_networks.data.datasets import TransformDataset
import torch
import numpy as np


def gen_dataset(config):
    """
    Generate a TransformDataset from a config dictionary with the following
    structure:
    config = {
        "pattern": {"type": obj, "params": {}},
        "transforms": {
            "0": {"type": obj, "params": {}},
            "1": {"type": obj, "params": {}}
         }
    }
    The "type" parameter in each dictionary specifies an uninstantiated dataset
    or transform class. The "params" parameter specifies a dictionary containing
    the keyword arguments needed to instantiate the class.
    """
    if "seed" in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
    # Catch for datasets and transforms that have no parameters
    if "params" not in config["pattern"]:
        config["pattern"]["params"] = {}
    for t in config["transforms"]:
        if "params" not in config["transforms"][t]:
            config["transforms"][t]["params"] = {}
            
    # Instantiate pattern object
    pattern = config["pattern"]["type"](**config["pattern"]["params"])
    
    # Instantiate transform objects
    transforms = [
        config["transforms"][k]["type"](**config["transforms"][k]["params"])
        for k in sorted(config["transforms"])
    ]
    
    # Generate dataset
    dataset = TransformDataset(pattern, transforms)
    return dataset
