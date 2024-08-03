
import time
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import shap
from lime import lime_tabular
from tqdm import tqdm
from interruptingcow import timeout

from dataset import DatasetCommons


from pickup_demand_prediction.models import SingleCellFCNN

class SizeAdaptableDuelingDQN(nn.Module):
    """Oriented on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    but with different state representation: expects two input maps, the (predicted) demand and supply as well as
    the taxi position."""

    def __init__(self, device, grid_size, action_space):
        super(SizeAdaptableDuelingDQN, self).__init__()
        self.device = device
        self.grid_size = grid_size
        self.action_space = action_space

        if self.grid_size == 10: 
            self.linear_size = 512
        elif self.grid_size == 20: 
            self.linear_size = 9216
        elif self.grid_size == 40: 
            self.linear_size = 57600
        elif self.grid_size == 80:
            self.linear_size = 175232

        self.conv1 = nn.Conv2d(2, 16, (5, 5), (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.conv6 = nn.Conv2d(64, 64, (3, 3), (1, 1))
        self.f1 = nn.Linear(self.linear_size + 2, 2048)  # 128 * 7 * 7 + 2 # + 2 for taxi position, 5*5=25 for action
        self.f2 = nn.Linear(2048, 1024)  # 128 * 7 * 7 + 2 # + 2 for taxi position, 5*5=25 for action
        self.value = nn.Linear(1024, 1)
        self.adv = nn.Linear(1024, self.action_space**2)

    def forward(self, x):  # x[0] is pick-up data, x[1] drop-off data, and x[2] the taxi position
        pu_do = torch.cat((x[0].to(self.device), x[1].to(self.device)), 1)
        pu_do = F.leaky_relu(self.conv1(pu_do))
        pu_do = F.leaky_relu(self.conv2(pu_do))
        if self.grid_size == 20:
            pu_do = F.leaky_relu(self.conv3(pu_do))
        elif self.grid_size == 40:
            pu_do = F.leaky_relu(self.conv3(pu_do))
            pu_do = F.leaky_relu(self.conv4(pu_do))
        elif self.grid_size == 200:
            pu_do = F.leaky_relu(self.conv3(pu_do))
            pu_do = F.leaky_relu(self.conv4(pu_do))
            pu_do = F.leaky_relu(self.conv5(pu_do))
            pu_do = F.leaky_relu(self.conv6(pu_do))

        r = torch.cat((pu_do.flatten(start_dim=1), x[2].to(self.device)), dim=1)
        r = F.relu(self.f1(r))
        r = F.relu(self.f2(r))
        value = F.leaky_relu(self.value(r))
        adv = F.leaky_relu(self.adv(r))
        adv_mean = torch.mean(adv, dim=1, keepdims=True) # type: ignore
        Q_values = value + adv - adv_mean
        return Q_values

class AdaptableCNN(nn.Module):
    """Based on the paper https://arxiv.org/abs/2010.01755."""

    def __init__(self):
        super(AdaptableCNN, self).__init__()
        p = [int(kernel_size / 2) for kernel_size in [3, 5, 7]]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=p[0])
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=p[1])
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=p[2])

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x.squeeze(dim=1)  # Removes 'channel' dimension

    def predict(self, x, *args):
        return self(x)

def compute_index(arr, alpha=.75, mean=11.8585) -> float:
    nof_requests, nof_taxis = arr[0], arr[1]
    return alpha * nof_requests / nof_taxis + (1 - alpha) * nof_requests / mean

def explain_via_index(r_y_p, t_y):
    start = time.process_time()
    result = np.stack((r_y_p.detach().numpy(), t_y.detach().numpy()), axis=2).reshape(city_size**2, 2)
    result = np.apply_along_axis(compute_index, axis=1, arr=result).reshape(city_size, city_size)
    return (time.process_time() - start)

def explain_via_arrows(r_y_p, t_y):
    positions = torch.tensor([[x, y] for y in range(city_size) for x in range(city_size)])
    r_y_p_repeated = r_y_p.repeat(positions.shape[0], 1, 1, 1)
    t_y_repeated = t_y.repeat(positions.shape[0], 1, 1, 1)
    
    start = time.process_time()
    q_values = repositioner((r_y_p_repeated, t_y_repeated, positions))
    actions = q_values.max(1)[1]
    return time.process_time() - start, q_values
    
def explain_via_uncertainty(r_y_p, t_y, q_values):
    start = time.process_time()
    min_q_value = q_values.min()
    max_q_value = q_values.max()
    return time.process_time() - start

def explain_request_estimator_via_shap(r_y_p, t_y):
    # Compute SHAP for request prediction and measure required time
    background_data = torch.randn(500, 20)
    explainer = shap.DeepExplainer(request_estimator, data=background_data)
    random_single_cell_input = torch.randn((1, 20))
    random_location = torch.randn(1, 2)
    start = time.process_time()
    for i in range(6):
        q_values = repositioner((r_y_p, t_y, random_location))
        action = q_values.max(1)[1]
        shap_values = explainer.shap_values(random_single_cell_input)
        shap_values = abs(shap_values[0])
        shap_values.sort()
        result = shap_values[:6]
    return time.process_time() - start

def explain_via_lime():
    # Compute LIME for whole model and measure the required time
    background_data = torch.randn((100, 5*city_size**2)).numpy()
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=background_data, 
        mode='regression',
        verbose=False,
        discretize_continuous=True,
        sample_around_instance=False,
        random_state=42
    )

    def wrapper(x_flat: np.ndarray):
        n_samples = x_flat.shape[0]
        x_flat = torch.from_numpy(x_flat).float()
        x_1 = x_flat[:, city_size**2*4:city_size**2*5].reshape(n_samples, 1, city_size, city_size)
        x_2 = torch.tensor(pos).repeat(n_samples, 1)
        requests_X_single_cell = torch.randn((n_samples * city_size**2, 20))
        r_y_p = request_estimator(requests_X_single_cell).reshape(n_samples, 1, city_size, city_size)
        return repositioner((r_y_p, x_1, x_2))[:, 0].detach().numpy()

    x = torch.randn((5 * city_size**2)).numpy()

    start = time.process_time()
    exp = explainer.explain_instance(x, wrapper, num_samples=1000, num_features=5 * city_size**2) 
    exp_local = exp.local_exp[0]
    exp_local.sort()
    return time.process_time() - start


class ComposedModel(nn.Module):

    def __init__(self, x_cells, y_cells, predictor, policy):
        super(ComposedModel, self).__init__()
        self.predictor = predictor
        self.policy = policy
        self.location = torch.tensor([1.0, 2.0])
        self.water_capacitry = torch.tensor([1.0])
        self.x_cells, self.y_cells = x_cells, y_cells
        self.s = self.x_cells*self.y_cells

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        n_samples = x.shape[0]
        demand = x[:,:self.s*4].reshape(n_samples, 4, self.x_cells, self.y_cells)
        indices = x[:,self.s*4:self.s*6].reshape(n_samples, 2, self.x_cells, self.y_cells)
        pois = x[:,self.s*6:self.s*7].reshape(n_samples, self.x_cells, self.y_cells)
        additional_features = x[:,self.s*6:self.s*6+13].reshape(n_samples, 13)
        supply = x[:,self.s*7+13:-2].reshape(n_samples, self.x_cells, self.y_cells)
        location = x[:,-2:]
        predictor_data = DatasetCommons.transform_to_single_cell_data(demand, pois, additional_features, supply, normalized=False, size=self.x_cells)
        predicted = self.predictor(torch.tensor(predictor_data)[:,:-1]).reshape(n_samples, 1, self.x_cells, self.y_cells)
        return self.policy((predicted, supply.reshape(n_samples, 1, self.x_cells, self.y_cells), location))



class SHAPExplainer():

    def __init__(self, x_cells, y_cells, model, nof_features) -> None:
        self.model = model
        self.x_cells, self.y_cells = x_cells, y_cells
        self.nof_features = nof_features
        self.background_size = 250
        self.explainer = shap.KernelExplainer(self._f, self._get_background_data())

    def explain(self, obs):
        shap_values = self.explainer.shap_values(obs.numpy(), nsamples=self.nof_features)
        return shap_values

    def _f(self, obs):
        q_values = self.model(obs).detach()
        action = q_values.argmax(1)
        return action.numpy()
    
    def _get_background_data(self):
        return shap.kmeans(np.random.random((1000, self.nof_features)), 30)


def measure_execution_time(explainer, obs, max_time=120):
    try:
        with timeout(max_time, exception=RuntimeError):
            start_time = time.time()
            explanation = explainer.explain(obs)
            elapsed_time = time.time() - start_time
    except RuntimeError:
        print(f'Explanations took more than {max_time}s.')
        elapsed_time, explanation = 121.0, (None, None)
    return elapsed_time, explanation


def main():
    warnings.filterwarnings("ignore")

    # Recreate explanation code - in a simplified version
    city_sizes = [20, 40, 80]  # 1000m, 500m, 250m, 125m
    n_samples = 10

    result = []

    for city_size in (cs_pbar := tqdm(city_sizes)):
        cs_pbar.set_description(f'CS: {city_size}')
        predictor = SingleCellFCNN()
        repositioner = SizeAdaptableDuelingDQN('cpu', city_size, 5)
        model = ComposedModel(city_size, city_size, predictor, repositioner)
        nof_params_request_estimator = sum(p.numel() for p in predictor.parameters())
        nof_params_repositioner = sum(p.numel() for p in repositioner.parameters())

        nof_features = city_size**2 * 8 + 2 + 13
        shap_explainer = SHAPExplainer(city_size, city_size, model, nof_features)
        shap_time = []

        print(f'CS: {city_size}; RE: {nof_params_request_estimator}; RP: {nof_params_repositioner}; SUM: {nof_params_request_estimator + nof_params_repositioner}')

        for sample in (sample_pbar := tqdm(list(range(n_samples)), leave=False)):

            obs = torch.randn((1, nof_features), dtype=torch.float32)
            elapsed_time, _ = measure_execution_time(shap_explainer, obs)
            shap_time.append(elapsed_time)
            print(f'{elapsed_time:.2f}')

        shap_time = np.asarray(shap_time)

        print(f'Env size: {city_size}')
        print(f'shap_time M: {shap_time.mean():.2f}; shap_time STD: {shap_time.std():.2f}')


if __name__ == "__main__":
    main()
