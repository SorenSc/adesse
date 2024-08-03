import numpy as np
import shap
import torch

from explanation.explainer import RepositioningAgentExplainer


class ShapExplainer(RepositioningAgentExplainer):

    def __init__(self, pu_y_p, do_y, model, taxi_pos):
        ppu = pu_y_p.reshape((do_y.shape[0], 1, 20, 20))
        do = do_y.reshape((do_y.shape[0], 1, 20, 20))
        background = torch.cat((ppu, do), axis=1)
        self.taxi_pos = taxi_pos

        def wrapper(flat_state):
            flat_state = torch.tensor(flat_state, dtype=torch.float32)
            flat_state = flat_state.reshape(flat_state.shape[0], 2, 20, 20)
            ppu, do = flat_state[:, 0, :, :].unsqueeze(dim=1), flat_state[:, 1, :, :].unsqueeze(dim=1)
            return model((ppu, do, self.taxi_pos.repeat(flat_state.shape[0], 1))).detach().numpy()

        self.explainer = shap.KernelExplainer(wrapper, shap.kmeans(background.flatten(start_dim=1), 30))

    def explain(self, input, model):
        # TODO: Is the taxi position handeled correctly? Randomization in the background is one option ...
        self.taxi_pos = input.taxis[0].location
        state = torch.cat((input.predicted_pickup_demand, input.dropoff_demand), dim=1).flatten(start_dim=1).numpy()
        shap_values = self.explainer.shap_values(state, nsamples=1000)
        shap_values = np.asarray(shap_values).reshape((25, 2, 20, 20))
        print(f'SHAP sum: {abs(shap_values).sum()}, SHAP min: {shap_values.min()}, SHAP max: {shap_values.max()}')
        return shap_values


