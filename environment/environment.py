import random

import gym
import numpy as np
import torch
from gym import Env
from gym.spaces import Discrete, Box
from torchvision import transforms

from datamodel.datamodel import Taxi, Action, RPState
from dataset import DatasetCommons
from logs.commons import setup_logger

environment_logger = setup_logger('environment_logger', './logs/environment.log')


class EnvironmentCommons:

    @staticmethod
    def single_to_two_dimensional_action(action: torch.Tensor, device='cpu'):
        """
        Transforms a single dimensional action into a two-dimensional one. For instance, the action with number two
        refers to (0, 2) or an movement towards the top of the environment. The first value represents the action on the
        x-axis and the second value the action on the y-axis.
        :param device:
        :param action: Something like tensor([[22]]); value has to be between 0 and 24.
        :return: (movement in x-direction, movement in y-direction)
        """
        return torch.tensor([[action % 5 - 2, (torch.div(action, 5, rounding_mode='trunc') - 2) * -1]]).to(device)

    @staticmethod
    def two_to_single_dimensional_action(action: torch.Tensor):
        a = action.clone().detach()
        return torch.tensor([(a[0] + 2) + (a[1] * -1 + 2) * 5], dtype=torch.int64)


class RepositioningEnvironment(Env):

    def __init__(self, data_loader, config, n_taxis=1, idx=None):
        self.action_space = Discrete(25)
        self.observation_space = Box(-50, 50, [20, 20])
        self.dl = data_loader
        self.n_taxis = n_taxis
        self.config = config
        self.random_dropoff_locations = DatasetCommons.create_set_of_random_dropoff_locations(None, size=None, load=True).to(self.config.dev)
        self.reset(idx)
        self.requests_y, self.requests_y_p, self.taxis_y = None, None, None

    @staticmethod
    def taxi_to_action(taxi):
        return Action(taxi.id, torch.Tensor(taxi.movement).reshape(1, 2))

    @staticmethod
    def initialize_taxis(n_taxis=1, device='cpu') -> {}:
        result = {}
        for taxi in range(n_taxis):
            result[taxi] = Taxi(
                taxi,
                torch.randint(low=0, high=20, size=(1, 2)).to(device),
                torch.tensor([[999, 999]]).to(device),
                torch.tensor([[999, 999]]).to(device),
                0
            )
        return result

    def get_new_taxi_position(self, action: Action):
        """Transform current to new location by making action two-dimensional and adding it to the current location."""
        two_dim_action = EnvironmentCommons.single_to_two_dimensional_action(action.movement, self.config.dev)
        new_location = self.state.taxis[action.taxi_id].location + two_dim_action
        new_location_clamped = torch.clamp(new_location, 0, 19)
        return new_location_clamped, two_dim_action

    def get_demand_supply_delta_at_location(self, t, new_pos_of_taxi):
        """
        It is important to reward the system based on the real and not the predicted demand supply delta as otherwise
        the prediction would represent the reality.
        :param t: Idx of current time step.
        :param new_pos_of_taxi: Something like tensor([[9, 1]]) or (x-position, y-position).
        :return:
        """
        pickup_demand = self.requests_y[t, new_pos_of_taxi[0][1], new_pos_of_taxi[0][0]].item()
        taxi_supply = self.taxis_y[t, new_pos_of_taxi[0][1], new_pos_of_taxi[0][0]].item() + 1  # Our taxi
        delta = pickup_demand - taxi_supply
        return pickup_demand, taxi_supply, delta

    @staticmethod
    def determine_reward(config, pickup_demand, taxi_supply, delta, movement):
        if delta >= 2:
            return config.reward_pickup_two_passengers
        elif delta == 1:
            return config.reward_pickup_one_passenger
        elif (delta < 1) & (pickup_demand > 0):  # Opportunity to get a trip even though delta is < 1
            if torch.rand((1)) < (pickup_demand / taxi_supply):  # Compete against other taxis and get a trip
                return config.reward_pickup_one_passenger
            else:  # Don't get a trip
                if movement:
                    return config.reward_no_pickup_but_movement
                elif not movement:
                    return config.reward_no_pickup_no_movement
        elif movement:
            return config.reward_no_pickup_but_movement
        elif not movement:
            return config.reward_no_pickup_no_movement
        else:
            environment_logger.error('The reward could not be determined.')
            return None

    def get_reward(self, two_dim_action, t, new_pos_of_taxi):
        pickup_demand, taxi_supply, delta = self.get_demand_supply_delta_at_location(t, new_pos_of_taxi)
        movement = not torch.all(torch.eq(two_dim_action, torch.tensor([[0, 0]]).to(self.config.dev))).item()
        reward = self.determine_reward(self.config, pickup_demand, taxi_supply, delta, movement)
        return reward

    def get_dropoff_location(self, size=1000):
        return self.random_dropoff_locations[torch.randint(low=0, high=size, size=(1, 1))[0]]

    def step(self, action: Action):
        """
        @param action: Movement in x and y direction. Both are at maximum 2.
        @return: The next state, a reward, and whether the run is done.
        """
        self.remaining_episode_length -= 1
        current_t = self.config.episode_length - self.remaining_episode_length
        nof_pick_ups, nof_passengers = 0, 0  # The latter is set to zero for pick-up of none or no passenger

        # Perform action for taxi or figure out its new position and collect reward
        pre_taxi_pos = self.state.taxis[action.taxi_id].location
        new_taxi_pos, two_dim_action = self.get_new_taxi_position(action)
        reward = self.get_reward(two_dim_action, current_t, new_taxi_pos)

        # Prepare next state and done to be returned
        if reward == self.config.reward_pickup_two_passengers:  # Pick up of two passengers
            # Don't do `self.remaining_episode_length -= 1` or alike as we do this step for the pick-up of two
            #   passengers in the simulator.
            future_data_index = current_t + 2  # Taxi driver needs two time steps to drop-off the passengers.
            new_taxi_pos = self.get_dropoff_location()
            nof_passengers, nof_pick_ups = 1, 2  # The former is set to one as we assume serving him/her takes one step
        elif reward == self.config.reward_pickup_one_passenger:  # Pick up of one passenger
            future_data_index = current_t + 1  # Taxi driver needs one time step to drop-off the passengers.
            new_taxi_pos = self.get_dropoff_location()
            nof_pick_ups = 1
        else:
            future_data_index = current_t + 1

        # Change next delta by current position of taxi; the predicted demand supply delta is used instead of the real
        # one as the actions shall be chosen based on the prediction rather than the real demand supply delta.
        self.taxis_y[future_data_index][new_taxi_pos[0][1], new_taxi_pos[0][0]] += 1  # taxi will be there

        self.state = RPState(
            self.requests_y_p[current_t + 1].reshape(1, 1, 20, 20),
            self.taxis_y[current_t + 1].reshape(1, 1, 20, 20),
            {0: Taxi(0, new_taxi_pos, torch.tensor([[999, 999]]), torch.tensor([[999, 999]]), nof_passengers)}
        )

        done = True if self.remaining_episode_length <= 1 else False

        environment_logger.info((f'T: {current_t}; REL: {self.remaining_episode_length}; '
                                 f'PP: {pre_taxi_pos[0].tolist()}; '
                                 f'A: {two_dim_action[0].tolist()}; '
                                 f'NP: {list(np.asarray(new_taxi_pos[0].cpu()))}; '
                                 f'R: {reward.tolist()[0]}'))

        return self.state, reward, done, {'nof_pick_ups': nof_pick_ups, 'current_t': current_t}

    def render(self):
        raise NotImplementedError

    def reset(self, idx=None):
        if idx is None:
            _, _, _, _, _, _, self.requests_y, self.requests_y_p, _, self.taxis_y = next(iter(self.dl))
        else:
            idx_list = [idx + j for j in list(range(self.config.episode_length + 3))]
            requests_y, requests_y_p, taxis_y = [], [], []
            for idx in idx_list:
                _, _, _, _, _, _, tmp_r_y, tmp_r_y_p, _, tmp_t_y = self.dl.dataset.__getitem__(idx)
                requests_y.append(tmp_r_y), requests_y_p.append(tmp_r_y_p), taxis_y.append(tmp_t_y)
            self.requests_y, self.requests_y_p, self.taxis_y = torch.stack(requests_y), torch.stack(requests_y_p), torch.stack(taxis_y)

        self.remaining_episode_length = self.config.episode_length
        self.state = RPState(
            self.requests_y_p[1].reshape(1, 1, 20, 20),
            self.taxis_y[1].reshape(1, 1, 20, 20),
            self.initialize_taxis(self.n_taxis, self.config.dev)
        )
        return self.state
