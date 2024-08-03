from torch import nn
from torch.nn import functional as F


class SingleCellFCNN(nn.Module):

    def __init__(self):
        super(SingleCellFCNN, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return x

    def predict(self, x, *args):
        return self(x)


class HaliemetalCNNNet(nn.Module):
    """Based on the paper https://arxiv.org/abs/2010.01755."""

    def __init__(self, config: {}):
        super(HaliemetalCNNNet, self).__init__()
        nofn = config.nof_neurons_per_layer
        ks = config.kernel_sizes
        p = [int(kernel_size / 2) for kernel_size in ks]
        self.conv1 = nn.Conv2d(in_channels=config.nof_input_maps, out_channels=nofn[0], kernel_size=ks[0],
                               padding=p[0])
        self.conv2 = nn.Conv2d(in_channels=nofn[0], out_channels=nofn[1], kernel_size=ks[1], padding=p[1])
        self.conv3 = nn.Conv2d(in_channels=nofn[1], out_channels=1, kernel_size=ks[2], padding=p[2])

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x.squeeze(dim=1)  # Removes 'channel' dimension

    def predict(self, x, *args):
        return self(x)