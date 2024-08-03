from collections import namedtuple
from dataclasses import dataclass

import torch


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


@dataclass
class Taxi:
    id: int  # identifier
    location: torch.float32  # x- and y-index of grid
    normed_location: torch.float32  # normed x- and y-index of grid
    destination: torch.float32  # x- and y-index of grid
    nof_passengers: int  # idle or occupied


@dataclass
class Action:
    """
    Action for one taxi.
    """
    taxi_id: int  # Taxi identifier
    movement: torch.float32  # Two-dimensional action


@dataclass
class RPState:
    """
    Repositioning state
    """
    predicted_pickup_demand: torch.float32
    dropoff_demand: torch.float32
    taxis: {}  # Taxis in the environment
