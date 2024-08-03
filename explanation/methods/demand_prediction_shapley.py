import calendar
from collections import OrderedDict
import logging

import numpy as np
import shap
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DatasetCommons
from environment.environment import EnvironmentCommons
from explanation.explainer import DemandPredictionAgentExplainer
from explanation.explanation_commons import ExplanationCommons


class DPShapExplainer(DemandPredictionAgentExplainer):

    def __init__(self, config,ds, model):

        self.config = config
        self.ds = ds
        self.model = model
        background_data = self.__create_background_data((50, 10)) 
        logging.warning('The normalization values are hard-coded in two places.')
        self.feature_to_unit = { 
            # feature name, unit, mean, std
            'x-index': ('SPECIAL', 0.0, 1.0),  # Not normalized
            'y-index': ('SPECIAL', 0.0, 1.0),  # Not normalized
            '#requests 30 minutes ago': ('SPECIAL', 11.858476560141144, 16.399040080276734),
            '#requests 20 minutes ago': ('SPECIAL', 11.858476560141144, 16.399040080276734),
            '#requests 10 minutes ago': ('SPECIAL', 11.858476560141144, 16.399040080276734),
            '#requests now': ('SPECIAL', 11.858476560141144, 16.399040080276734),
            '#POI': ('SPECIAL', 12.42, 13.6322),
            'hour': ('SPECIAL', 11.5, 6.922186552431729),
            'minute': ('SPECIAL', 29.5, 17.318102282486574),
            'weekday': ('SPECIAL', 3.0, 2.0),
            'month': ('SPECIAL', 5.5, 3.452052529534663),
            'temperature': ('°C', 12.59857542610023, 10.289749748716927),
            'wind': ('km/h', 9.03183668277792, 6.506661862899048),
            'humidity': ('%', 56.25228949376749, 18.279020770525968),
            'barometer': ('hPa', 1017.0959425082676, 7.802860124425204),
            'view': ('km', 14.523289239379293, 3.356261603888417),
            'snow': ('cm', 0.01469091834138896, 0.1203124900402099),
            'precipitation': ('l/m^2', 0.05737725769524294, 0.23256205192295804),
            'cloudy': ('SPECIAL', 0.35105571101500893, 0.47730032346391243),
            'holiday': ('SPECIAL', 0.02930552022386161, 0.16866151519617759),
        }
        means = [v[1] for v in self.feature_to_unit.values()]
        stds = [v[2] for v in self.feature_to_unit.values()]
        self.model_n = nn.Sequential(OrderedDict([
            ('normalize', NormalizationLayer(means, stds)),
            ('model', self.model)
        ]))
        self.explainer = shap.DeepExplainer(self.model_n, data=background_data)
    

    def explain(self, time_step, locations):
        _, _, _, r_X, r_pois, r_X_add, r_y, _, _, _ = self.ds.__getitem__(time_step)
        
        # Non-normalized values
        single_cell_Xy = DatasetCommons.transform_to_single_cell_data(
            r_X.reshape(1, 4, 20, 20), r_pois.reshape(1, 20, 20),
            r_X_add.reshape(1, 13), r_y.reshape(1, 20, 20))
        single_cell_X_r = single_cell_Xy.reshape(20, 20, 21)[:20]

        result = []
        for location in locations:
            shap_values_at_location = self.explainer.shap_values(
                torch.tensor(single_cell_X_r[location[1], location[0], :20].reshape(1, 20)))
            X_at_location = single_cell_X_r[location[1], location[0]]
            sorted_feature_importance = sorted(list(zip(list(self.feature_to_unit.keys()), X_at_location,
                shap_values_at_location[0])), key=lambda tup: abs(tup[2]))
            most_important_features = [[v[0], self.value_to_text(v), round(v[2], 2)] for v in sorted_feature_importance[-6:]]
            most_important_features.reverse()
            result.append(most_important_features)

        return result

    def value_to_text(self, e):
        """Transfer value into text. Normally this is done by adding the unit to the value, but weekday, cloudy, and
        holiday are managed differently."""
        if self.feature_to_unit[e[0]][0] == '':
            return f'{e[1]:.2f}'
        elif self.feature_to_unit[e[0]][0] == 'SPECIAL':
            if e[0] == 'weekday':
                return list(calendar.day_name)[int(e[1])]
            elif e[0] == 'cloudy':
                return 'cloudy' if int(e[1]) == 0 else 'not cloudy'
            elif e[0] == 'holiday':
                return 'holiday' if int(e[1]) == 0 else 'no holiday'
            elif e[0].startswith('#requests'):
                return str(int(e[1])) + ' trips'
            elif e[0] in ['x-index', 'y-index', '#POI', 'hour', 'minute', 'month']:
                return str(int(e[1]))
        else:
            return f'{e[1]:.2f} {self.feature_to_unit[e[0]][0]}'

    def explain_via_table(self, dp_shap_values):
        sorted_feature_importance = sorted(list(zip(list(self.feature_to_unit.keys()), dp_shap_values[1], dp_shap_values[0])),  # feature values, feature importance
                                           key=lambda tup: tup[2])
        return [[v[0], self.value_to_text(v), round(v[2], 2)] for v in sorted_feature_importance[:6]]

    def get_upcoming_locations(self, simulator):
        """
        Suppose you are only changing the location of the state you are currently in and then
        exploit the policy to move through the space, then this function returns up to five next
        locations plus the current location. Once the locations are collected, duplicates are 
        removed and the remaining locations are returned.
        :param dev:
        :param simulator:
        :param two_d_action:
        :param net:
        :return:
        """
        locations = []
        locations.append(simulator.env.state.taxis[0].location.tolist()[0])
        for i in range(5):
            _, action_ahead, _ = ExplanationCommons.get_q_values_and_action_if_exploited(
                simulator, 0, torch.tensor(np.array([locations[-1]])).to(self.config.dev))
            two_d_action_ahead = EnvironmentCommons.single_to_two_dimensional_action(action_ahead).tolist()[0]
            locations.append([locations[-1][0] + two_d_action_ahead[0], locations[-1][1] + two_d_action_ahead[1]])
        return [list(e) for e in list(dict.fromkeys([tuple(l) for l in locations]))]

    def __create_background_data(self, bg_size):
        _, _, _, r_X, r_pois, r_X_add, r_y, _, _, _ = \
            next(iter(DataLoader(self.ds, batch_size=bg_size[0], shuffle=True)))
        single_cell_X = DatasetCommons.transform_to_single_cell_data(r_X, r_pois, r_X_add, r_y)[:, :20]
        random_indices = np.random.choice(single_cell_X.shape[0], np.prod(bg_size), replace=False)
        return torch.tensor(single_cell_X[random_indices])


class NormalizationLayer(nn.Module):
    """
    This layer normalizes the output of the AdditionalFeatureDataset transformed into the single 
    cell format. 
    """

    def __init__(self, means, stds):
        super().__init__()
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

    def forward(self, x):
        return torch.divide(torch.subtract(x, self.means), self.stds)
        # return torch.add(torch.mul(x, self.stds), self.means)  # Anti-normalization