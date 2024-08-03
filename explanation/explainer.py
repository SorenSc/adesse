from abc import ABC, abstractmethod

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

raw_x_indices = torch.tensor([v % 5 - 2 for v in list(range(25))]).reshape((5, 5))
raw_y_indices = torch.flip(torch.tensor([v // 5 - 2 for v in list(range(25))]).reshape((5, 5)), dims=(0, 1))

mpl.rcParams['figure.dpi'] = 300
font = 'Consolas'


def normalize_shap_value(value, max_value=450):
    """Min-max normalization with minimum = 0 and maximum = `max_value`."""
    if value > 0:
        return abs(value) / max_value / 2 + 0.5
    elif value < 0:
        return abs(value) / max_value / 2
    else:
        return 0.5


class RepositioningAgentExplainer(ABC):

    @abstractmethod
    def explain(self):
        """
        Generate an explanation with the corresponding XAI-method.
        :return:
        """
        raise NotImplementedError

    def visualize_matrix(self, state_included, matrix, file_name, taxi_pos, proposed_action, title, cmap,
                         explanation=None, min_diff=None, max_diff=None, next_locations=None, ppu=True):

        if state_included:
            matrix = np.float32(matrix[0, 0])  # Convert to float to enable the next step
        else:
            matrix = np.zeros((20, 20))

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        hm = sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=0, vmax=65, square=True, cbar=False)

        if self.name == 'ActionMap':
            for a in explanation:
                alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
                hm.arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], alpha=alpha, head_width=0.1)
        elif self.name == 'SARFA':
            explanation_cmap = plt.get_cmap('Purples')
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if ppu:
                        explanation_color = explanation_cmap(explanation[i, j, 0] * 2)
                    else:
                        explanation_color = explanation_cmap(explanation[i, j, 1] * 2)
                    hm.add_patch(
                        Rectangle((i + 0.05, j + 0.05), 0.95, 0.95, fill=False, edgecolor=explanation_color, lw=1))
        elif self.name == 'SHAP':
            explanation_cmap = plt.get_cmap('bwr')
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if ppu:
                        explanation_color = explanation_cmap(normalize_shap_value(explanation[0, i, j]))
                    else:
                        explanation_color = explanation_cmap(normalize_shap_value(explanation[1, i, j]))
                    hm.add_patch(
                        Rectangle((i + 0.05, j + 0.05), 0.95, 0.95, fill=False, edgecolor=explanation_color, lw=1))

        if next_locations != '':
            index = 0
            for l in next_locations:
                if index > 1:
                    hm.annotate(chr(ord('A') + index - 2), (l[0] + 0.45, l[1] + 0.45), color='royalblue', fontsize=12,
                                ha='center', va='center', font=font, weight='bold')
                index += 1

        hm.set_ylim((0, matrix.shape[0]))
        if title is not None: plt.title(title, fontname=font)
        plt.xlabel('x-index of grid', fontname=font), plt.ylabel('y-index of grid', fontname=font)
        plt.yticks(fontname=font, rotation=0), plt.xticks(fontname=font)

        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))

        if self.name == 'ActionMap':
            cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys')), shrink=0.6)
            cb.outline.set_visible(False)
            cb.set_ticks([0, 1.0])
            cb.set_ticklabels(['0', '1'], font=font)
            cb.set_label('Arrow colors; 0 is uncertain and 1 certain', font=font)
        elif self.name == 'SARFA':
            cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=explanation_cmap), shrink=0.6)
            cb.outline.set_visible(False)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.set_ticklabels(['0', '0.125', '0.25', '0.375', '0.5'], font=font)
            cb.set_label('SARFA values; 0 is no and 1 strong influence', font=font)
        elif self.name == 'SHAP':
            # When the shap values of a single action are used, an appropriate maximum/minimum value is 125/-125.
            # When the shap values of a all actions are used, an appropriate maximum/minimum value is 450/-450.
            cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=explanation_cmap), shrink=0.6)
            cb.outline.set_visible(False)
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cb.set_ticklabels(['-450', '-225', '0', '225', '450'], font=font)
            cb.set_label('SHAP values; -450 is strong neg. and 450 strong pos. influence', font=font)

        if state_included:
            cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap)), shrink=0.6)
            cb.outline.set_visible(False)
            label_values = [0, 10, 20, 30, 40, 50, 60]
            cb.set_ticks([i / 65 for i in label_values])
            cb.set_ticklabels([str(i) for i in label_values], font=font)
            # cb.set_label('Demand values', font=font)

        if file_name is not None:
            plt.savefig(f'./explanation/graphics/{file_name}', bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_explanation(self, state_included, plot_titles, taxi_pos, predicted_pickup_demand, dropoff_demand,
                              explanation, current_t, proposed_action, next_locations, game=False):
        """
        When the shap values of a single action are used, an appropriate maximum/minimum value is 125/-125.
        When the shap values of a all actions are used, an appropriate maximum/minimum value is 450/-450.
        :return:
        """
        file_names = self.get_file_names(current_t, game)
        min_diff, max_diff = self.get_differences(explanation)
        figure_ppu = self.visualize_matrix(
            state_included, predicted_pickup_demand, file_names[0], taxi_pos, proposed_action, plot_titles[0], 'Greens',
            explanation, min_diff, max_diff, next_locations)
        figure_dp = self.visualize_matrix(
            state_included, dropoff_demand, file_names[1], taxi_pos, proposed_action, plot_titles[1], 'PuRd',
            explanation, min_diff, max_diff, next_locations, False)
        return figure_ppu, figure_dp

    def get_file_names(self, current_t=0, game=False):
        if game:
            return None, None
        else:
            return (f'state_r_hat/{current_t}.PNG',
                    f'state_t/{current_t}.PNG')

    def get_differences(self, explanation):
        if self.name == 'ActionMap':
            return np.asarray([a[4] for a in explanation]).min(), np.asarray([a[4] for a in explanation]).max()
        else:
            return None, None


class DemandPredictionAgentExplainer(ABC):

    @abstractmethod
    def explain(self):
        raise NotImplementedError
