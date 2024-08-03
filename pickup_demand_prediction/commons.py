import pathlib

import numpy as np
import torch
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter


def mean_relative_error(predicted_values: np.ndarray, target_values: np.ndarray):
    """
    The calculation of the MRE is debatable. We are orienting on the version for instance mentioned by
    https://doi.org/10.1145/3293317. An alternative definition is given for instance by
    https://doi.org/10.1007/s42421-020-00030-z. Using the second definition is more correct from my gut feeling, but,
    then, using the MAPE doesn't make sense anymore. Therefore, we decided for the former version.
    """
    return sum(abs(target_values - predicted_values)) / sum(target_values)


def evaluate_performance(y, y_p, criterion_mse, criterion_mae, identifier='train'):
    """Generate dictionary of (error metric name, error value) for given train prediction."""
    train_y_r_d, train_y_p_r_d = (y.cpu().detach().numpy().flatten(),
                                  y_p.cpu().detach().numpy().flatten())

    return {
        f'MAE/{identifier}': criterion_mae(y_p, y).cpu().detach().numpy().tolist(),
        f'MRE/{identifier}': mean_relative_error(train_y_p_r_d, train_y_r_d),
        f'MSE/{identifier}': criterion_mse(y_p, y).cpu().detach().numpy().tolist(),
        f'RMSE/{identifier}': torch.sqrt(criterion_mse(y_p, y)).cpu().detach().numpy().tolist(),
    }


def save_model(epoch, model, optimizer, criterion, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, save_path)


def count_parameters(model):
    # Modified version of
    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    # print(table)
    print(f"#trainable params: {total_params}")
    return total_params


def create_summary_writer(model_name):
    runs_folder = './pickup_demand_prediction/results/runs/'
    pathlib.Path(runs_folder).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(runs_folder + model_name + '/')
    return writer







