import copy
import logging
import math
import random
from datetime import datetime
import dacite
import toml
import torch

from config.commons import enhance_config
from config.config import GameConfig, RepositioningConfiguration
from datamodel.datamodel import Action
from environment.environment import EnvironmentCommons
from explanation.explainers import Baselinev2Explainer, ComposedExplainer
from explanation.explanation_commons import ExplanationCommons
from game.commons import get_impossible_actions_as_single_str
from simulation import Simulator


class GameToSimulatorConnector():
    """Connection between game and simulator."""

    def __init__(self, composed) -> None:
        self.config_requests, self.config_taxis, self.config_exp = self.get_configs(composed)
        self.simulator = self.setup_simulator()
        if self.config_exp.setting == 'composed':
            self.explainer = ComposedExplainer(self.simulator, self.config_requests)
        elif self.config_exp.setting == 'baseline':
            self.explainer = Baselinev2Explainer(self.simulator)
        else:
            raise NotImplementedError
        self.last_location = None

    def step(self, run_id, action, scenario, idx, taxi_x, taxi_y):
        """Perform one step in game and the simulator."""
        if action[0].item() == 99:  # Reset
            self.last_location = None
            if scenario == 'A':
                self.scenario = scenario 
                self.idx, self.accumulated_reward, self.rp_exp_selection = 629, 0, '1'
                next_state = self.simulator.env.reset(self.idx)
                self.simulator.env.state.taxis[0].location = torch.tensor([[3, 4]])
                info, reward = {'current_t': 0}, torch.tensor(0)
            elif scenario == 'B':
                self.scenario = scenario 
                self.idx, self.accumulated_reward, self.rp_exp_selection = idx, 0, '1'
                next_state = self.simulator.env.reset(self.idx)
                self.simulator.env.state.taxis[0].location = torch.tensor([[taxi_x, taxi_y]])
                info, reward = {'current_t': 0}, torch.tensor(0)
            else:
                raise NotImplementedError
        else:  # Normal step
            self.last_location = self.simulator.env.state.taxis[0].location.tolist()[0]
            selected_action = EnvironmentCommons.two_to_single_dimensional_action(torch.tensor(action))
            next_state, reward, done, info = self.simulator.env.step(Action(0, selected_action))

        _, proposed_action_1d, action_filter = ExplanationCommons.get_q_values_and_action_if_exploited(self.simulator)
        proposed_action = (next_state.taxis[0].location + EnvironmentCommons.single_to_two_dimensional_action(proposed_action_1d)).tolist()[0]
        taxi_location = next_state.taxis[0].location.tolist()[0]
        current_idx = self.idx + info['current_t']
        self.accumulated_reward += reward.item()
        impossible_actions_str = get_impossible_actions_as_single_str(action_filter)
        explanation = self.explainer.explain(self.idx, proposed_action_1d.item())

        with open('./logs/log.csv', 'a') as f:
            f.write(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")},{self.config_exp.setting},{run_id},' + 
                f'{self.scenario},{current_idx},{reward.item()},{self.accumulated_reward},' + 
                f'{taxi_location[0]},{taxi_location[1]},{action.tolist()[0]},' + 
                f'{action.tolist()[1]},{proposed_action[0]},{proposed_action[1]},' + 
                f'{self.rp_exp_selection} \n'
            )

        # To mark which locations where accessed by explainee for demand prediction explanation
        self.rp_exp_selection = '1'

        return {
            'last_location': self.last_location,
            'taxi_location': taxi_location,
            'proposed_action': proposed_action,
            'current_idx': current_idx,
            'last_reward': reward.item(),
            'accumulated_reward': self.accumulated_reward,
            'impossible_actions': impossible_actions_str,
            'explanation': explanation,
        }
    
    def reset(self, run_id, scenario='A'):
        """Reset the simulator and game."""
        self.scenario = scenario
        if scenario == 'A':
            self.idx = 624
            self.simulator.env.reset(self.idx)
            self.simulator.env.state.taxis[0].location = torch.tensor([[3, 17]])
            self.accumulated_reward = 0
            self.rp_exp_selection = '1'
        else:
            raise NotImplementedError

    def setup_simulator(self):
        """
        Sets up the simulator. Therefore, the reproducability is created via setting random seeds,
        and a  simulator object with the trained neural networks.
        """
        logging.info('GameToSimulatorConnector: Setting up the Simulator...')

        random.seed(self.config_exp.random_seed)
        torch.manual_seed(self.config_exp.random_seed)

        logging.info('GameToSimulatorConnector: Generating simulator in which the game is played..')
        simulator = Simulator(self.config_requests, self.config_taxis, None, self.config_exp.load, 
            self.config_exp.single_cell)

        logging.info('GameToSimulatorConnector: Loading networks of simulator...')
        from dqn.network import DuelingDQN
        simulator.policy_net = DuelingDQN(self.config_exp.device)
        simulator.policy_net.load_state_dict(torch.load('./models/' + self.config_exp.repositioner, 
                map_location=torch.device(self.config_exp.device)))
        simulator.policy_net.eval()
        for param in simulator.policy_net.parameters():
            param.requires_grad = False
        return simulator

    @staticmethod
    def get_configs(composed):
        """
        Creates three config dictionaries - one for the requests, one for the taxis, and one for the
        explanations.
        """

        # Requests config
        config_requests = dacite.from_dict(
                data_class=RepositioningConfiguration,
                data=toml.load(r'./config/config.toml'))
        config_requests = enhance_config(config_requests)
        config_requests.demand_dict_file = f'./data/yellow_tripdata_{int(math.sqrt(config_requests.grid_cell_area))}_{config_requests.time_bin_size}.npy'

        # Taxis config
        config_taxis = copy.copy(config_requests)
        config_taxis.demand_dict_file = f'./data/yellow_tripdata_{int(math.sqrt(config_taxis.grid_cell_area))}_{config_taxis.time_bin_size}_do.npy'

        # Explanation config
        config_file = r'./config/explanation_composed.toml' if composed else r'./config/explanation_baseline.toml'
        config_exp = dacite.from_dict(
            data_class=GameConfig,
            data=toml.load(config_file)
        )

        return config_requests, config_taxis, config_exp

    def manage_request_estimation_exp_selection(self, selection):
        self.rp_exp_selection = self.rp_exp_selection + str(selection)
