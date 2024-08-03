import random
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from datamodel.datamodel import Transition


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DuelingDQN(nn.Module):
    """Oriented on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    but with different state representation: expects two input maps, the (predicted) demand and supply as well as
    the taxi position."""

    def __init__(self, device):
        super(DuelingDQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(2, 16, (5, 5), (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.f1 = nn.Linear(64 * 12 * 12 + 2, 2048)  # 128 * 7 * 7 + 2 # + 2 for taxi position, 5*5=25 for action
        self.f2 = nn.Linear(2048, 1024)  # 128 * 7 * 7 + 2 # + 2 for taxi position, 5*5=25 for action
        self.value = nn.Linear(1024, 1)
        self.adv = nn.Linear(1024, 25)

    def forward(self, x):  # x[0] is pick-up data, x[1] drop-off data, and x[2] the taxi position
        pu_do = torch.cat((x[0].to(self.device), x[1].to(self.device)), 1)
        pu_do = F.leaky_relu(self.conv1(pu_do))
        pu_do = F.leaky_relu(self.conv2(pu_do))
        pu_do = F.leaky_relu(self.conv3(pu_do))
        r = torch.cat((pu_do.flatten(start_dim=1), x[2].to(self.device)), dim=1)
        r = F.relu(self.f1(r))
        r = F.relu(self.f2(r))
        value = F.leaky_relu(self.value(r))
        adv = F.leaky_relu(self.adv(r))
        adv_mean = torch.mean(adv, dim=1, keepdims=True)
        Q_values = value + adv - adv_mean
        return Q_values
