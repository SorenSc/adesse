import logging
import math
import random
from itertools import count

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config.config import RepositioningConfiguration

from datamodel.datamodel import Transition, RPState, Action
from dataset import DatasetCommons
from dqn.network import ReplayMemory, DuelingDQN
from environment.environment import RepositioningEnvironment, EnvironmentCommons
from logs.commons import setup_logger

simulation_logger = setup_logger('simulation_logger', './logs/simulation.log')


class Simulator:

    def __init__(
        self, 
        config_requests: RepositioningConfiguration, 
        config_taxis: RepositioningConfiguration, 
        idx: int | None = None, 
        load: bool = False, 
        single_cell: bool = False
    ):
        self.config_requests = config_requests
        self.config_taxis = config_taxis

        logging.info('Simulator: Creating data loaders...')
        dl = DatasetCommons.create_data_loader(config_requests, config_taxis, load, single_cell)

        logging.info('Simulator: Initializing policy and target network...')
        self.policy_net = DuelingDQN(self.config_requests.dev).to(self.config_requests.dev)
        self.target_net = DuelingDQN(self.config_requests.dev).to(self.config_requests.dev)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        logging.info('Simulator: Preparing optimizer, memory, environment, and some other things...')
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.config_requests.learning_rate)
        self.memory = ReplayMemory(self.config_requests.replay_memory_size)
        self.steps_done = 0
        self.env = RepositioningEnvironment(dl, self.config_requests, 1, idx)

        self.raw_x_indices = torch.tensor([v % 5 - 2 for v in list(range(25))]).reshape((5, 5)).to(self.config_requests.dev)
        self.raw_y_indices = torch.flip(torch.tensor([v // 5 - 2 for v in list(range(25))]).reshape((5, 5)), dims=(0, 1)).to(self.config_requests.dev)
        self.actions = torch.tensor(list(range(25))).to(self.config_requests.dev)

        self.writer = SummaryWriter('./runs/' + self.config_requests.model_name + '/')
        self.criterion = nn.SmoothL1Loss()


    def create_filter_for_valid_actions(self, location):
        x_indices, y_indices = self.raw_x_indices + location[0][0], self.raw_y_indices + location[0][1]
        action_filter = (((x_indices <= 19) & (x_indices >= 0)) & ((y_indices <= 19) & (y_indices >= 0))).flatten()
        return action_filter

    def compute_threshold(self):
        """Computes threshold of the decayed esilon greedy strategy."""
        return self.config_requests.eps_end + (self.config_requests.eps_start - self.config_requests.eps_end) * \
               math.exp(-1. * self.steps_done / self.config_requests.eps_decay)

    def select_action(self, state: RPState, taxi_id: int, eps_threshold=None):
        q_values = None
        if eps_threshold is None: eps_threshold = self.compute_threshold()  # if enables usage for exploitation
        self.steps_done += 1
        if random.random() > eps_threshold:  # If random value is larger than threshold, policy will be exploited
            with torch.no_grad():
                location = state.taxis[taxi_id].location
                action_filter = self.create_filter_for_valid_actions(location)
                q_values = self.policy_net((state.predicted_pickup_demand, state.dropoff_demand, state.taxis[taxi_id].location))
                q_values[0][~action_filter] = -5  # Make invalid actions manually unattractive
                action = torch.argmax(q_values, keepdim=True)  # Select action with maximal q-value
        else:  # or a valid random action is taken
            action_filter = self.create_filter_for_valid_actions(state.taxis[taxi_id].location)
            valid_actions = self.actions[action_filter]
            action = valid_actions[torch.randint(0, len(valid_actions), (1, 1))].to(self.config_requests.dev)
        return Action(taxi_id, action), q_values

    def optimize_model(self):
        if len(self.memory) < self.config_requests.batch_size:
            return
        transitions = self.memory.sample(self.config_requests.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for  detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one
        # after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.config_requests.dev,
                                      dtype=torch.bool)

        # Compute the non-final next states
        next_pu_p_batch = torch.cat([s.predicted_pickup_demand for s in batch.next_state if s is not None])
        next_do_batch = torch.cat([s.dropoff_demand for s in batch.next_state if s is not None])
        next_pos_batch = torch.cat([s.taxis[0].location for s in batch.next_state if s is not None])
        non_final_next_states = (next_pu_p_batch, next_do_batch, next_pos_batch)

        # Derive state, action and reward from batch
        pu_p_batch = torch.cat([s.predicted_pickup_demand for s in batch.state])
        do_batch = torch.cat([s.dropoff_demand for s in batch.state])
        pos_batch = torch.cat([s.taxis[0].location for s in batch.state])
        state_batch = (pu_p_batch, do_batch, pos_batch)
        action_batch = torch.cat([t.movement for t in batch.action])
        reward_batch = torch.cat(batch.reward)

        # Dueling DQN
        with torch.no_grad():
            policy_Q_next = torch.zeros((self.config_requests.batch_size, 25), device=self.config_requests.dev)
            policy_Q_next[non_final_mask] = self.policy_net(non_final_next_states)
            policy_max_action = torch.argmax(policy_Q_next, dim=1, keepdim=True)
            target_Q_next = torch.zeros((self.config_requests.batch_size, 25), device=self.config_requests.dev)
            target_Q_next[non_final_mask] = self.target_net(non_final_next_states)
            expected_state_action_values = self.config_requests.gamma * target_Q_next.gather(1, policy_max_action) + reward_batch.reshape(128, 1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        loss = self.criterion(state_action_values, expected_state_action_values)  # Comparison

        self.optimizer.zero_grad()  # Otherwise the gradients would be accumulated over subsequent backward passes
        loss.backward()  # Allows to call .backward() multiple times 
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Clamp parameters between [-1, 1]
        self.optimizer.step()

    def train(self):
        for i_episode in (pbar := tqdm(range(self.config_requests.n_episodes))):
            rewards, nof_pick_ups = [], 0
            state = self.env.reset()
            for t in count():
                previous_taxi_position = state.taxis[0].location
                if state.taxis[0].nof_passengers < 1:  # Taxi is idle
                    action, _ = self.select_action(state, 0)  # Select and perform an action
                    next_state, reward, done, info = self.env.step(action)
                    if done: next_state = None
                    self.memory.push(state, action, next_state, reward)
                    self.optimize_model()  # Perform one step of the optimization (on the policy network)
                    if done:
                        nof_pick_ups += info['nof_pick_ups']
                        self.track_status_at_episode_end(rewards, nof_pick_ups, i_episode, state,
                                                    previous_taxi_position, action, t)
                        break
                    state = next_state  # Move to the next state

                    # Track performance
                    rewards.append(reward[0])
                    nof_pick_ups += info['nof_pick_ups']

                else:  # Taxi is not idle
                    state.taxis[0].nof_passengers -= 1
                    self.env.remaining_episode_length -= 1  # The show must go on (or the time :))

                pbar.set_description(f'E: [{i_episode+1}:{self.config_requests.n_episodes}] T: [{t + 1}], steps_done:{self.steps_done}')

            if (i_episode < 10) or (i_episode % 15 == 0):
                self.add_net_weights_to_summary_writer(i_episode)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.config_requests.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (i_episode % 500 == 0) & self.config_requests.evaluate_model:
                torch.save(self.target_net.state_dict(), './models/' + self.config_requests.model_name + '_' + str(i_episode))

    def track_status_at_episode_end(self, rewards, nof_pick_ups, i_episode, state, previous_taxi_position, action, t):
        avg_reward = torch.tensor(rewards, dtype=torch.float16).mean().numpy()
        self.writer.add_scalar('#pick-ups/train', nof_pick_ups, i_episode)
        self.writer.add_scalar('Reward/train', avg_reward, i_episode)
        self.writer.add_scalar('DEpsiolonGThreshold/train', self.compute_threshold(), i_episode)

        taxi_position = state.taxis[0].location.cpu().tolist()[0]
        message = (f'SIM: E: [{i_episode + 1}:{self.config_requests.n_episodes}], '
                   f'T: {t + 1}; '
                   f'REL: {self.env.remaining_episode_length}; '
                   f'P: {list(np.asarray(previous_taxi_position[0].cpu()))}; '
                   f'A: {EnvironmentCommons.single_to_two_dimensional_action(action.movement)[0].tolist()}; '
                   f'NP: {taxi_position}; '
                   f'NofPU: {nof_pick_ups}; '
                   f'R: {avg_reward:{4}.{2}f}; {np.asarray(torch.tensor(rewards, dtype=torch.float16))[:10]}')
        simulation_logger.info(message)

    def add_net_weights_to_summary_writer(self, t):
        self.writer.add_histogram('conv1/weight', self.policy_net.conv1.weight, global_step=t)
        self.writer.add_histogram('conv1/bias', self.policy_net.conv1.bias, global_step=t)
        self.writer.add_histogram('conv2/weight', self.policy_net.conv2.weight, global_step=t)
        self.writer.add_histogram('conv2/bias', self.policy_net.conv2.bias, global_step=t)
        self.writer.add_histogram('conv3/weight', self.policy_net.conv3.weight, global_step=t)
        self.writer.add_histogram('conv3/bias', self.policy_net.conv3.bias, global_step=t)
        self.writer.add_histogram('fc1/weight', self.policy_net.f1.weight, global_step=t)
        self.writer.add_histogram('fc1/bias', self.policy_net.f1.bias, global_step=t)
        self.writer.add_histogram('value/weight', self.policy_net.f1.weight, global_step=t)
        self.writer.add_histogram('value/bias', self.policy_net.f1.bias, global_step=t)
        self.writer.add_histogram('adv/weight', self.policy_net.f1.weight, global_step=t)
        self.writer.add_histogram('adv/bias', self.policy_net.f1.bias, global_step=t)

    def exploit_policy(self, net, location=None):
        """
        Exploit a policy - represented by a neural network - and collect the states and rewards.
        @param net:
        @return:
        """

        # Initialize the environment
        self.env.reset()

        if location is not None:
            self.env.state.taxis[0].location = location

        result = []
        for t in count():  # Iterate until episode ends
            q_values = np.zeros((5, 5))
            time_point = self.env.start_idx + self.config_requests.episode_length - self.env.remaining_episode_length
            if self.env.state.taxis[0].nof_passengers < 1:  # Taxi is idle
                location = self.env.state.taxis[0].location
                action_filter = self.create_filter_for_valid_actions(location)
                q_values = net((self.env.state.predicted_pickup_demand, self.env.state.dropoff_demand,
                                            self.env.state.taxis[0].location))
                q_values[0][~action_filter] = q_values.min() - 1
                action = q_values.max(1)[1].view(1, 1)
                state_location = self.env.state.taxis[0].location[0].numpy()
                next_state, reward, done, _ = self.env.step(Action(0, action))
                q_values = q_values.detach().numpy().reshape(5, 5)  # Simply for reporting
            else:  # Taxi is not idle
                self.env.state.taxis[0].nof_passengers -= 1
                self.env.remaining_episode_length -= 1  # The show must go on (or the time :))
                state_location = self.env.state.taxis[0].location[0].numpy()
                action = torch.tensor([[12]])  # Do nothing
                reward = 0

            result.append({
                't': t,
                'actual_time': time_point,
                'actual_time_readable': DatasetCommons.idx_to_datetime(time_point, self.config_requests).
                    strftime("%Y-%m-%d - %A, %H:%M"),
                'actual_delta': (self.env.pu_y - self.env.do_y)[time_point].numpy(),
                'predicted_delta': (self.env.pu_y_p - self.env.do_y)[time_point].numpy(),
                'q-values': q_values,
                '#pickups': self.env.pu_y[time_point].numpy().sum(),
                '#predicted_pickups': self.env.pu_y_p[time_point].numpy().sum(),
                '#dropoffs': self.env.do_y[time_point].numpy().sum(),
                'taxi_position': state_location,
                'action': EnvironmentCommons.single_to_two_dimensional_action(action)[0].tolist(),
                'reward': reward,
            })

            if done: self.env.state = None
            if done: break

        return result

