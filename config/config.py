from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class GameConfig:
    device: str  # Device used for the neural network models
    random_seed: int  # To make results reproducable
    setting: str  # Use composed or baseline setting
    load: bool  # Load previously pickled dataloader instead of creating it
    repositioner: str  # Name of file that contains trained neural network for the repositioner
    request_predictor: str  # Name of file that contains trained neural network for the reques_predictor
    single_cell: bool  # Single cell or city-wide model for request prediction?


@dataclass
class RepositioningConfiguration:

    # Dataset ###
    # Device used to train/run the neural networks on
    dev: str

    # Start and end point of the data---normally the first and last datetime in the dataset
    start: datetime
    end: datetime

    # Number of time-bins cut off from the end of the dataset to be used for testing the final
    # models
    test_size: int

    # Percentage from dataset_size - test_size used for validating along the training
    validation_size: float

    # Number of input maps---each relates to a time-bin---passed into a demand prediction model
    nof_input_maps: int

    # Size of a time-bin in minutes; when set to 30, a day has 48 time-bins.
    time_bin_size: int

    # Time-shift refers to not only start from `start` and moving in steps of `time_bin_size`
    # towards `end` but doing the same from for instance `start` + 5 minutes, `start` + 10 minutes,
    # etc. to artificially increase the dataset size.
    time_shift: bool

    # Pickup and dropoff distribution are loaded from a `.npy` file. This field specifies the end
    # of the file name as it sometimes changes depending on the version of the previous data
    # preparation and the considered dataset.
    file_name_ext: str
    model_name_ext: str

    # Pseudo grid ###
    # The grid used to discretize locations is a 'pseudo-grid', meaning that it not really contains
    # all cells, but simply the x and y separation points. The following four values specify the
    # pseudo-grids (abbreviated via 'pgc') bottom left (bl) and top right (tr) coordinates of the
    # grid.
    pgc_grid_bl_lat: float
    pgc_grid_bl_lon: float
    pgc_grid_tr_lat: float
    pgc_grid_tr_lon: float

    # We specify the grid cell area instead of the side length to be able to handle grid types
    # beyond square grids. The area refers to the height * width, both in meters.
    grid_cell_area: int

    # Selection of area to for instance reduce the area considered to the city center. The
    # abbreviation `red_t` stands for 'reduction from top'.
    red_t: int
    red_b: int
    red_l: int
    red_r: int

    # We tried to use an input with a smaller grid cell size than the output size for demand
    # prediction. It didn't work, but that is the reason why we have this configuration parameter.
    finer_input: bool

    # DataLoader ###
    batch_size: int

    # Tracking of results
    evaluate_model: bool

    # Reinforcement learning ###
    learning_rate: float
    gamma: float
    eps_start: float  # Starting probability for performing a random action
    eps_end: float  # Ending probability for performing a random action
    eps_decay: int  # Rate of decay
    target_update: int
    n_episodes: int
    episode_length: int  # Length of an episode; for instance 7*48=336 for a week
    replay_memory_size: int  # Size of replay memory

    reward_pickup_two_passengers: int
    reward_pickup_one_passenger: int
    reward_no_pickup_but_movement: int
    reward_no_pickup_no_movement: int

    # Evaluation ###
    n_runs: int  # Number of runs for which the policy is exploited for visualization

    # Demand Prediction ###
    dataset_len: int
    holidays = None
    weather_file = None
    demand_dict_file = None
    random_seed: int
    X_mean: float
    X_std: float
    normalized: bool


@dataclass
class WholeCityDemandPredictionConfiguration:
    batch_size: int
    demand_dict_file: str
    dev: str
    evaluate_model: bool
    grid_cell_area: int
    kernel_sizes: List[int]
    learning_rate: float
    model_name: str
    n_epochs: int
    nof_input_maps: int
    nof_neurons_per_layer: List[int]
    normalized: bool
    pgc_grid_bl_lat: float
    pgc_grid_bl_lon: float
    pgc_grid_tr_lat: float
    pgc_grid_tr_lon: float
    random_seed: int
    red_b: int
    red_l: int
    red_r: int
    red_t: int
    start: datetime
    end: datetime
    time_bin_size: int
    weight_decay: float
    X_mean: float
    X_std: float


@dataclass
class SingleCellDemandPredictionConfiguration:
    batch_size_train: int
    batch_size_train_v2: int
    batch_size_valid: int
    demand_dict_file: str
    dev: str
    evaluate_model: bool
    grid_cell_area: int
    learning_rate: float
    model_name: str
    n_epochs: int
    nof_input_maps: int
    nof_neurons_per_layer: List[int]
    normalized: bool
    pgc_grid_bl_lat: float
    pgc_grid_bl_lon: float
    pgc_grid_tr_lat: float
    pgc_grid_tr_lon: float
    random_seed: int
    red_b: int
    red_l: int
    red_r: int
    red_t: int
    start: datetime
    end: datetime
    time_bin_size: int
    weight_decay: float
    weather_file: str
    X_mean: float
    X_std: float
