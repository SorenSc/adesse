
from abc import ABC, abstractmethod
import logging
import random
from typing import List
import numpy as np
import torch
from dataset import DatasetCommons
from explanation.methods.demand_prediction_shapley import DPShapExplainer
from lime import lime_tabular

from explanation.methods.indices import WeightedRequestTaxiIndex
from explanation.methods.rp_action_map import ActionMapExplainer


class IExplainer(ABC):
    """
    :param explanation: Dict of raw explanations so that it can be easily passed through.
    """

    def __init__(self, explanation=None) -> None:
        super().__init__()
        self.explanation = explanation

    @abstractmethod
    def explain(self, idx: int, action: List[int] | None):
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        raise NotImplementedError


class ComposedExplainer(IExplainer):

    def __init__(self, simulator=None, config=None, load=False) -> None:
        super().__init__()
        logging.info('ComposedExplainer: Setting it up...')
        self.simulator = simulator
        self.explanation = {
            'index': None,
            'arrows': None,
            'table': None
        }
        logging.info('ComposedExplainer: Building explainers...')
        self.state_index_explainer = WeightedRequestTaxiIndex(.75, 11.8585)
        self.arrow_explainer = ActionMapExplainer()
        self.request_explainer = DPShapExplainer(
                config = config,
                ds = self.simulator.env.dl.dataset,
                model = simulator.env.dl.dataset.request_predictor
        )

    def explain(self, idx, action):
        self.explanation['index'] = self.__get_index().tolist()
        self.explanation['arrows'] = self.arrow_explainer.explain(self.simulator, 'ToMin').tolist()
        self.explanation['locations'] = self.request_explainer.get_upcoming_locations(self.simulator)
        self.explanation['table'] = self.request_explainer.explain(idx, self.explanation['locations'])
        return self.explanation

    def __get_index(self):
        rh = self.simulator.env.state.predicted_pickup_demand[0, 0].numpy()
        rh = np.clip(rh, 0, rh.max())
        t = self.simulator.env.state.dropoff_demand[0, 0].numpy()
        result = np.stack((rh, t), axis=2).reshape(20 * 20, 2)
        result = np.apply_along_axis(self.state_index_explainer.compute, axis=1, arr=result).reshape(20, 20)
        # Remove nan and infinity to enable visualization
        result = np.nan_to_num(result, nan=0, posinf=0)
        return result

    def visualize(self):
        raise NotImplementedError


class Baselinev2Explainer(IExplainer):

    def __init__(self, simulator) -> None:
        super().__init__()
        self.simulator = simulator
        self.background_data = self.__create_background_data()
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.background_data, 
            mode='regression',
            verbose=True,
            discretize_continuous=True,
            sample_around_instance=False, 
            random_state=42
        )

    def explain(self, idx, action):

        logging.warning('Could the values be so high, bcs the normalization is somehow messed up?')

        r_X_n, r_pois_n, r_X_add_n, r_X, r_pois, r_X_add, r_y, r_y_p, t_X, t_y = self.simulator.env.dl.dataset.__getitem__(idx)
        x0 = r_X_n.flatten()
        x1 = t_y.flatten()
        x = torch.cat((x0, x1)).numpy()

        pos = self.simulator.env.state.taxis[0].location.tolist()[0]

        def wrapper(x_flat: np.ndarray):
            n_samples = x_flat.shape[0]
            x_flat = torch.from_numpy(x_flat).float()
            x_1 = x_flat[:, 1600:2000].reshape(n_samples, 1, 20, 20)
            x_2 = torch.tensor(pos).repeat(n_samples, 1)
            requests_X_single_cell, _ = DatasetCommons.prepare_data_fcnn(
                x_flat[:, :1600].reshape(n_samples, 4, 20, 20),
                r_pois_n.flatten().repeat(n_samples, 1).reshape((n_samples, 20, 20)),
                r_X_add_n.repeat(n_samples, 1),
                r_y.flatten().repeat(n_samples, 1).reshape((n_samples, 20, 20)),
                'cpu', False, 0, True)
            r_y_p = self.simulator.env.dl.dataset.request_predictor(requests_X_single_cell).reshape(n_samples, 1, 20, 20)
            return self.simulator.policy_net((r_y_p, x_1, x_2))[:, action].numpy()

        exp = self.explainer.explain_instance(x, wrapper, num_samples=1000, num_features=2000) 
        exp_local = exp.local_exp[1]
        exp_local.sort()
        return {
            'r_X': r_X.tolist(), 
            't_y': t_y.tolist(), 
            'lime': np.asarray([e[1] for e in exp_local]).reshape(5, 20, 20).tolist()
        }

    def __create_background_data(self, size=25):
        # Idea: Get idx that are around the idx requested to create more 'local explanations'
        idx_background = []
        while len(idx_background) < size:
            random_idx = random.choice(self.simulator.env.dl.dataset.idx)
            random_idx_as_datetime = DatasetCommons.transform_idx_to_time(random_idx, self.simulator.config_requests)
            if (int(random_idx_as_datetime.strftime('%w')) in [1, 2, 3, 4]) and (int(random_idx_as_datetime.strftime('%H')) in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]): 
                idx_background.append(random_idx)

        result = []
        for idx in idx_background:
            r_X_n_s, _, _, _, _, _, _, _, _, t_y_s = self.simulator.env.dl.dataset.__getitem__(idx)  # s for sample
            result.append(torch.cat((r_X_n_s.flatten(), t_y_s.flatten())))
        return torch.cat(result).reshape(len(result), 2000).numpy()  # 4 x 20 x 20 + 20 x 20

    def visualize(self):
        raise NotImplementedError