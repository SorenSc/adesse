import random

import torch

from config.commons import get_configs
from logs.commons import clear_log_files
from simulation import Simulator

clear_log_files()


def main():

    config_requests, config_taxis = get_configs()
    random.seed(config_requests.random_seed)
    torch.manual_seed(config_requests.random_seed)

    simulator = Simulator(config_requests, config_taxis)
    simulator.train()


if __name__ == '__main__':
    main()
