import torch
import argparse
import hydra
from omegaconf import OmegaConf
from featup.train_implicit_upsampler import my_app

torch.manual_seed(1001)


with open("FeatUp/featup/configs/implicit_upsampler.yaml") as f:
    basic_conf_file = f.read()
basic_conf = OmegaConf.create(basic_conf_file)


@hydra.main(config_path="", config_name="")
def start_featup(split_n: int) -> None:
    torch.set_num_threads(20)
    new_config = basic_conf.copy()
    new_config.dataset = f"data/imagenet_reduced/splits/{split_n}"
    print(new_config)
    my_app(new_config)


parser = argparse.ArgumentParser(description="get split")
parser.add_argument(
    "--n",
    type=int,
    default=0,
    metavar="N",
)
if __name__ == "__main__":
    args = parser.parse_args()

    start_featup(args.n)
