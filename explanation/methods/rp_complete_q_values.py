import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import torch

from explanation.explainer import RepositioningAgentExplainer


class CompleteQValueExplainer(RepositioningAgentExplainer):

    def __init__(self):
        self.name = 'CompleteQValue'


    def explain(self, simulator, normalize=True, normalize_overall=True):
        explanation = np.zeros((100, 100))
        for pos_x in range(20):
            for pos_y in range(20):

                taxi_pos = torch.tensor([[pos_x, pos_y]])

                # Get the Q-values for the current state and given taxi position.
                q_values = simulator.policy_net((simulator.env.state.predicted_pickup_demand,
                                  simulator.env.state.dropoff_demand, taxi_pos))

                m_pos_x, m_pos_y = pos_x * 5, pos_y * 5

                if normalize and not normalize_overall:  # Normalize per action
                    action_filter = simulator.create_filter_for_valid_actions(taxi_pos)  # Not all actions are possible
                    qvmin, qvmax = q_values[0][action_filter].min(), q_values[0][action_filter].max()
                    q_values[0][~action_filter] = 0
                    q_values[0][action_filter] = (q_values[0][action_filter] - qvmin) / (qvmax - qvmin)

                explanation[m_pos_y:m_pos_y + 5, m_pos_x:m_pos_x + 5] = np.flip(q_values[0].detach().numpy().reshape(5, 5), axis=0)

        if normalize and normalize_overall:
            q_min, q_max = explanation[2:-2, 2:-2].min(), explanation[2:-2, 2:-2].max()
            explanation[2:-2, 2:-2] = (explanation[2:-2, 2:-2] - q_min) / (q_max - q_min)

        # Set borders to zero
        rv = 0 if normalize else explanation.min() - .1  # Replace value
        explanation[:2, :], explanation[:, :2], explanation[-2:, :], explanation[:, -2:] = rv, rv, rv, rv
        explanation[5, :], explanation[:, 5], explanation[94, :], explanation[:, 94] = rv, rv, rv, rv

        return explanation

    def visualize(self, matrix, file_name):
        cmap = sns.color_palette('magma', as_cmap=True)
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # ax.set_title('Normed Q-values')
        sns.heatmap(matrix, square=True, cmap=cmap, ax=ax, cbar_kws={"shrink": .75})
        ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        label_pos, labels = [float(p * 5 + 2.5) for p in range(20)], [str(p) for p in range(20)]
        ax.set_xticks(label_pos, labels), ax.set_yticks(label_pos, labels)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        for vlp in range(grid_size // 5): plt.vlines(vlp * 5, ymin=0, ymax=grid_size, color='black', linewidth=0.5)
        for hlp in range(grid_size // 5): plt.hlines(hlp * 5, xmin=0, xmax=grid_size, color='black', linewidth=0.5)

        cbar = ax.collections[0].colorbar
        label_values = cbar.get_ticks()
        cbar.set_ticks([i for i in label_values])
        cbar.set_ticklabels([f'{i:.1f}' for i in label_values], font=font)

        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

