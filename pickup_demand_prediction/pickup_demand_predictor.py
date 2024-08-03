import torch

from torch import nn
import torch.nn.functional as F


class FCNNet(nn.Module):

    def __init__(self, config: {}):
        super(FCNNet, self).__init__()
        self.height, self.width, self.nof_input_maps = 20, 20, config.nof_input_maps
        self.size = self.nof_input_maps * self.height * self.width
        self.config = config

        self.fc1 = nn.Linear(self.size, int(self.size*0.9))
        self.fc2 = nn.Linear(int(self.size*0.9), int(self.size*0.8))
        self.fc3 = nn.Linear(int(self.size*0.8), int(self.size*0.7))
        self.fc4 = nn.Linear(int(self.size*0.7), int(self.size*0.5))
        self.fc5 = nn.Linear(int(self.size*0.5), self.height * self.width)

    def forward(self, X):
        x = F.relu(self.fc1(X.flatten(start_dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x.reshape(X.shape[0], self.height, self.width)


class PickupDemandPredictor:

    @staticmethod
    def load_network(config):
        net = FCNNet(config)
        net.load_state_dict(torch.load('./pickup_demand_prediction/models/5487_m2-fcnn_500_4_10_256_001___'))
        net.to(config.dev)
        return net
