import random

import dacite
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import SingleCellDemandPredictionConfiguration
from dataset import NormalizedAdditionalFeatureDataset, DatasetCommons
from pickup_demand_prediction.commons import evaluate_performance, save_model, create_summary_writer
from pickup_demand_prediction.models import SingleCellFCNN
from pickup_demand_prediction.single_cell_catboost import get_dataset_length


def prepare_data_fcnn(X, X_pois, X_add, y, dev, randomized=False, size=1):
    data = DatasetCommons.transform_to_single_cell_data(X, X_pois, X_add, y, True)
    if randomized:
        indices = np.random.choice(data.shape[0], size, replace=False)
        data = data[indices]
    return torch.tensor(data[:, :20]).to(dev).float(), torch.tensor(data[:, 20].reshape(data.shape[0], 1)).to(dev).float()


def track_performance_via_writer(model, dl, train_val_str, len_train_dl, writer, config, epoch, i, crit_mse, crit_mae):
    X, X_pois, X_add, y = next(iter(dl))
    batch_size = X.shape[0]
    X, y = prepare_data_fcnn(X, X_pois, X_add, y, config.dev)
    y_p = model.predict(X)
    eval_res = evaluate_performance(y.reshape(batch_size, 20, 20), y_p.reshape(batch_size, 20, 20), crit_mse, crit_mae, train_val_str)
    [writer.add_scalar(key, value, epoch) for (key, value) in eval_res.items()]
    return eval_res


def run(config):
    config = dacite.from_dict(data_class=SingleCellDemandPredictionConfiguration, data=config)
    config.dataset_len = get_dataset_length(config)
    config.model_name = f'{config.model_name}_{config.batch_size_train}_{config.batch_size_train_v2}_{config.learning_rate}'
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    _, idx_train_dp, idx_valid_dp, idx_test_dp, _, _, _ = \
        DatasetCommons.get_train_validation_test_indices_for_repositioning(config)

    train_ds = NormalizedAdditionalFeatureDataset(config, idx_train_dp)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size_train, drop_last=True)
    valid_ds = NormalizedAdditionalFeatureDataset(config, idx_valid_dp)
    valid_dl = DataLoader(valid_ds, shuffle=True, batch_size=config.batch_size_valid, drop_last=True)

    model = SingleCellFCNN()
    model = model.to(config.dev)

    writer = create_summary_writer(config.model_name)
    crit_mse, crit_mae = nn.MSELoss(), nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    train_running_loss, counter, net_best_loss = 0.0, 0, float('inf')
    for epoch in (pbar := tqdm(range(config.n_epochs))):
        for i, (X, X_pois, X_add, y) in enumerate(train_dl):
            counter += 1
            X, y = prepare_data_fcnn(X, X_pois, X_add, y, config.dev, True, config.batch_size_train_v2)
            model.zero_grad()
            optimizer.zero_grad()
            y_p = model.predict(X)
            loss = crit_mse(y_p, y)
            current_loss = loss.item()
            train_running_loss += current_loss
            loss.backward()
            optimizer.step()

            pbar.set_description(f'Training; Epoch: [{epoch + 1}:{config.n_epochs}], Batch: [{i + 1}:{len(train_dl)}], '
                                 f'MSE: {current_loss:.2f}')

        track_performance_via_writer(model, train_dl, 'train', len(train_dl), writer, config, epoch, i, crit_mse, crit_mae)
        val_res = track_performance_via_writer(model, valid_dl, 'val', 1, writer, config, epoch, i, crit_mse, crit_mae)
        if val_res['MSE/val'] < net_best_loss:
            save_model(epoch, model, optimizer, crit_mse, './pickup_demand_prediction/results/models/' + config.model_name)
            net_best_loss = val_res['MSE/val']
