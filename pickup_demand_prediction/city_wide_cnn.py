import copy
import math
import pathlib
import random

import dacite
import numpy as np
import toml
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import WholeCityDemandPredictionConfiguration
from dataset import CityCenterDataset, DatasetCommons
from pickup_demand_prediction.commons import evaluate_performance, count_parameters, save_model, create_summary_writer
from pickup_demand_prediction.models import HaliemetalCNNNet


def train(config, epoch, model, data_loader, optimizer, crit_mse, crit_mae, device, writer=None, pbar=None):
    model.train()
    train_running_loss, scaled_running_loss, counter, step = 0.0, 0.0, 0, 0

    for i, (X, y) in enumerate(data_loader):
        counter += 1
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        y_p = model.predict(X)
        loss = crit_mse(y_p, y)
        current_loss = loss.item()
        train_running_loss += current_loss
        loss.backward()
        optimizer.step()

        eval_res = evaluate_performance(y, y_p, crit_mse, crit_mae)
        if pbar is not None:
            pbar.set_description(f'Training; Epoch: [{epoch + 1}:{config.n_epochs}], Batch: [{i + 1}:{len(data_loader)}], '
                                 f'RMSE: {eval_res["RMSE/train"]:.2f}, '
                                 f'MAE: {eval_res["MAE/train"]:.2f}, '
                                 f'MRE: {eval_res["MRE/train"]:.2f}')

        scaled_running_loss += eval_res["MSE/train"]
        step += 1

        if writer is not None:
            writer.add_scalar('MSEWithNoRescaling/train', current_loss, epoch * len(data_loader) + i)
            [writer.add_scalar(key, value, epoch * len(data_loader) + i) for (key, value) in eval_res.items()]

    return scaled_running_loss / counter


def validate(config, epoch, model, data_loader, crit_mse, crit_mae, device, writer=None):
    model.eval()
    valid_running_loss, scaled_running_loss, step = 0.0, 0.0, 0

    with torch.no_grad():
        (X, y) = next(iter(data_loader))
        X, y = X.to(device), y.to(device)
        y_p = model.predict(X)
        loss = crit_mse(y_p, y)
        valid_running_loss += loss.item()
        eval_res = evaluate_performance(y, y_p, crit_mse, crit_mae, 'val')
        scaled_running_loss += eval_res['MSE/val']
        step += 1

        if writer is not None:
            [writer.add_scalar(key, value, epoch * len(data_loader)) for (key, value) in eval_res.items()]

    return scaled_running_loss


def evaluate_model(config, model_name, model, valid_dl):
    crit_mse, crit_mae = nn.MSELoss(), nn.L1Loss()
    total_nof_trainable_params = count_parameters(model)
    # Enables running it with different nets with same configuration, like when selecting the best model

    result_array = []

    for i, (X, y) in (pbar := tqdm(enumerate(valid_dl))):
        if i < 25:  # Do at maximum 25 steps here to make evaluation faster
            X, y = X.to(torch.device(config.dev)), y.to(torch.device(config.dev))
            model.zero_grad()
            y_p = model.predict(X)
            eval_res = evaluate_performance(y, y_p, crit_mse, crit_mae, identifier='valid')
            result_array.append(eval_res)
            pbar.set_description(f'Batch: [{i}:{len(valid_dl)}], '
                                 f'RMSE/valid: {eval_res["RMSE/valid"]:.2f}, '
                                 f'MAE/valid: {eval_res["MAE/valid"]:.2f}, '
                                 f'MRE/valid: {eval_res["MRE/valid"]:.2f}')

    values = []
    for metric in ['RMSE/valid', 'MAE/valid', 'MRE/valid']:
        values.append(np.array([r[metric] for r in result_array]).sum() / len(result_array))
    pbar.set_description(f'RMSE/valid: {values[0]:.2f}, '
                         f'MAE/valid: {values[1]:.2f}, '
                         f'MRE/valid: {values[2]:.2f}')

    print(f"{model_name}, {int(math.sqrt(config.grid_cell_area))}, {config.time_bin_size}, "
                f"{config.nof_input_maps}, {values[0]}, {values[1]}, {values[2]}, "
                f"{total_nof_trainable_params} \n")

    with open('./results/results.csv', 'a+') as f:
        f.write(f"{model_name}, {int(math.sqrt(config.grid_cell_area))}, {config.time_bin_size}, "
                f"{config.nof_input_maps}, {values[0]}, {values[1]}, {values[2]}, "
                f"{total_nof_trainable_params} \n")

    return model_name, values[0], values[1], values[2]


def generate_and_train_nn_model(config: {}) -> (int, nn.Module):
    """Builds and trains a given neural network."""



def create_data_loaders(config, dataset_class):
    _, idx_train_dp, idx_valid_dp, idx_test_dp, _, _, _ = DatasetCommons.get_train_validation_test_indices_for_repositioning(config)
    train_dl = DataLoader(dataset_class(config, idx_train_dp),
                          shuffle=True, batch_size=config.batch_size, drop_last=True)
    valid_dl = DataLoader(dataset_class(config, idx_valid_dp),
                          shuffle=True, batch_size=config.batch_size, drop_last=True)
    test_dl = DataLoader(dataset_class(config, idx_test_dp),
                         shuffle=True, batch_size=config.batch_size, drop_last=True)
    return train_dl, valid_dl, test_dl


def run():
    config = toml.load(r'./config/whole_city_demand_predictor.toml')
    config = dacite.from_dict(data_class=WholeCityDemandPredictionConfiguration, data=config)
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    # generate_and_train_nn_model(config)

    writer = create_summary_writer(config.model_name)
    train_dl, valid_dl, test_dl = create_data_loaders(config, CityCenterDataset)
    X, y = next(iter(train_dl))
    X = X.to(torch.device(config.dev))
    model = HaliemetalCNNNet(config)
    model = model.to(config.dev)
    writer.add_graph(model, X)

    models_folder = "./pickup_demand_prediction/results/models/"
    pathlib.Path(models_folder).mkdir(parents=True, exist_ok=True)
    crit_mse, crit_mae = nn.MSELoss(), nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = torch.device(config.dev)
    net_best, net_best_loss = copy.deepcopy(model), float('inf')

    for epoch in (pbar := tqdm(range(config.n_epochs))):
        _ = train(
            config, epoch, model, train_dl, optimizer, crit_mse, crit_mae, device, writer, pbar)
        val_loss = validate(
            config, epoch, model, valid_dl, crit_mse, crit_mae, device, writer)

        if val_loss < net_best_loss:
            save_model(epoch, model, optimizer, crit_mse, models_folder + config.model_name)
            net_best_loss = val_loss

    if writer is not None:
        writer.close()  # Close summary writer after training
    if config.evaluate_model:
        evaluate_model(config, config.model_name, model, valid_dl)
