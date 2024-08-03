import torch


def load_network(device, model_name):
    model_name = './models/' + model_name

    from dqn.network import DuelingDQN
    net = DuelingDQN(device)
    net.load_state_dict(torch.load(model_name, map_location=torch.device(device)))

    net.eval()
    return net
