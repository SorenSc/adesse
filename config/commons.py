import copy
import math
import time

import dacite
import toml
import torch

from config.config import RepositioningConfiguration
from dataset import DatasetCommons


def load_toml_config(file_name):
    raw_config = toml.load(file_name)
    return dacite.from_dict(data_class=RepositioningConfiguration, data=raw_config)


def get_configs(filepath=None, game=False):
    if filepath == None:
        config = load_toml_config(r'./config/config.toml')
    else:
        config = load_toml_config(filepath)

    if game:
        # config.demand_dict_file = f'../../data/yellow_tripdata_2015-01_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}.npy'
        config.demand_dict_file = f'../../data/yellow_tripdata_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}.npy'
    else:
        config.demand_dict_file = f'./data/yellow_tripdata_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}.npy'
    config = enhance_config(config)
    do_config = copy.copy(config)
    if game:
        # do_config.demand_dict_file = f'../../data/dropoff_dict_yellow_tripdata_2015-01_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}_do.npy'
        do_config.demand_dict_file = f'../../data/yellow_tripdata_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}_do.npy'
    else:
        do_config.demand_dict_file = f'./data/yellow_tripdata_{int(math.sqrt(config.grid_cell_area))}_{config.time_bin_size}_do.npy'
    do_config.normalized = False
    return config, do_config


def enhance_config(config):
    """Loading the config file. This method also contains a couple of testing mechanisms to check the validity
    of the given config file as well as
    - adding the mean and standard deviation---used for normalization---to the
    config and
    - selecting the input dictionary corresponding to the config.

    You can get the mean and std values by `np.asarray(list(self.demand_dict.values())).mean()` and
    `np.asarray(list(self.demand_dict.values())).std()`
    """
    config.dataset_len = DatasetCommons.get_dataset_length(config)

    if (config.grid_cell_area == 250000) & (config.time_bin_size == 10):
        config.X_mean, config.X_std = 11.782582680405781, 16.43430422610831
    else:
        raise ValueError('No mean and std value for this grid cell size.')

    config.model_name = f'{str(int(time.time()))}_{config.model_name_ext}_'
    config.reward_pickup_two_passengers = torch.tensor(config.reward_pickup_two_passengers).reshape((1)).to(config.dev)
    config.reward_pickup_one_passenger = torch.tensor(config.reward_pickup_one_passenger).reshape((1)).to(config.dev)
    config.reward_no_pickup_but_movement = torch.tensor(config.reward_no_pickup_but_movement).reshape((1)).to(config.dev)
    config.reward_no_pickup_no_movement = torch.tensor(config.reward_no_pickup_no_movement).reshape((1)).to(config.dev)

    return config


def time_window_of_X_is_thirty_minutes(config):
    var_1 = (config.time_bin_size == 30) and (config.time_bin_scaling == 1) and \
            ('_30_' in config.demand_dict_input_file)
    var_2 = (config.time_bin_size == 30) and (config.time_bin_scaling == 6) and \
            ('_5_' in config.demand_dict_input_file)
    return var_1 or var_2


def load_config_via_toml(config_file: str, id, start=28):
    """Load config from toml-file."""
    config = toml.load(config_file)
    config.dataset_len = int((config.end - config.start).total_seconds() // 60 // config.time_bin_size -
                             config.nof_input_maps)
    config.model_name = '_'.join([
        str(id),  # Identifier
        config_file[start:-5],  # Used config file
        config.network_type,
        str(int(math.sqrt(config.grid_cell_area))),
        str(config.nof_input_maps),
        str(config.time_bin_size),
        str(config.batch_size),
        str(config.learning_rate)[2:],
    ])
    return config


def get_explanation_scenarios():
    return [
        {'name': 'SC1', 'hour': 4, 'minute': 0, 'weekday': 6, 'desc': 'SC1 - Sun. at 4 a.m.'},
        {'name': 'SC2', 'hour': 8, 'minute': 30, 'weekday': 6, 'desc': 'SC2 - Sun. at 8:30 a.m.'},
        {'name': 'SC3', 'hour': 17, 'minute': 0, 'weekday': 6, 'desc': 'SC3 - Sun. at 5 p.m.'},
        {'name': 'SC4', 'hour': 4, 'minute': 0, 'weekday': 2, 'desc': 'SC4 - Wed. at 4 a.m.'},
        {'name': 'SC5', 'hour': 8, 'minute': 30, 'weekday': 2, 'desc': 'SC5 - Wed. at 8:30 a.m.'},
        {'name': 'SC6', 'hour': 17, 'minute': 0, 'weekday': 2, 'desc': 'SC6 - Wed. at 5 p.m.'},
    ]