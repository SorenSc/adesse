import random

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch

from explanation.explainer import RepositioningAgentExplainer


class LIMEExplainer(RepositioningAgentExplainer):

    def __init__(self, pu_y_p, do_y, model):

        self.pu_y_p, self.do_y = pu_y_p, do_y

        self.model = model
        self.model.eval()

        self.training_data = self.create_training_data(10000)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data,
            mode='regression',
            verbose=True,
            discretize_continuous=True,
            sample_around_instance=False,
            random_state=42
        )

        self.p = np.random.randint(low=0, high=20, size=(2))
        self.a = 0


    def create_training_data(self, size=1000):
        random_idx = np.random.choice(np.arange(self.pu_y_p.shape[0]), size, replace=False)
        random_pu = self.pu_y_p[random_idx].flatten(start_dim=1).numpy()
        random_do = self.do_y[random_idx].flatten(start_dim=1).numpy()
        return np.hstack((random_pu, random_do))


    def explain(self, position, action):
        self.p, self.a = position, action

        def dqn_wrapper(flat_state):
            flat_state = torch.tensor(flat_state)
            nof_s = flat_state.shape[0]  # Number of samples
            d = flat_state[:, :400].reshape(nof_s, 1, 20, 20).to(torch.float32)
            s = flat_state[:, 400:800].reshape(nof_s, 1, 20, 20).to(torch.float32)
            p = np.stack((np.repeat(self.p[0], nof_s), (np.repeat(self.p[1], nof_s))), axis=1)
            return self.model((d, s, torch.tensor(p))).detach().numpy()[:, self.a]  # Only explain action 0

        exp = self.explainer.explain_instance(self.training_data[0], dqn_wrapper, num_samples=1000, num_features=800)
        exp_local = exp.local_exp[0]
        exp_local.sort()
        return np.asarray([e[1] for e in exp_local]).reshape((2, 20, 20))


    @staticmethod
    def to_2d():
        pass