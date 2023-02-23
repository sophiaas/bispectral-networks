import argparse
from bispectral_networks.trainer import run_trainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Name of .py config file with no extension.",
    default="translation_experiment",
)
parser.add_argument("--device", type=int, help="device to run on, -1 for cpu", default=-1)
parser.add_argument(
    "--n_examples", type=int, help="number of data examples", default=5e6
)
parser.add_argument("--seed", type=int, default=None)


args = parser.parse_args()
if args.device == -1:
    args.device = 'cpu'

print("Running experiment on device {}...".format(args.device))
exec("from configs.{} import master_config, logger_config".format(args.config))

def run_wrapper():
    run_trainer(
        master_config=master_config,
        logger_config=logger_config,
        device=args.device,
        n_examples=args.n_examples,
        seed=args.seed
    )

run_wrapper()
