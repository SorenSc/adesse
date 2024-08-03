import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from config import commons
from environment.environment import EnvironmentCommons


class ExplanationCommons:

    @staticmethod
    def rel_recommendation_to_absolute(simulator):
        q_values, recommendation = ExplanationCommons.get_q_values_and_action_if_exploited(simulator, 0)
        return (simulator.env.state.taxis[0].location +
                EnvironmentCommons.single_to_two_dimensional_action(recommendation)).tolist()[0]

    @staticmethod
    def load_configs():
        return commons.get_configs()

    @staticmethod
    def get_q_values_and_action_if_exploited(simulator, taxi_id=0, location=None):
        if location is None:
            location = simulator.env.state.taxis[taxi_id].location
        action_filter = simulator.create_filter_for_valid_actions(location)
        q_values = simulator.policy_net((simulator.env.state.predicted_pickup_demand,
                                         simulator.env.state.dropoff_demand, location))
        q_values[0][~action_filter] = -5  # Make invalid actions manually unattractive
        return q_values, q_values.max(1)[1].view(1, 1), action_filter  # Select action with maximal q-value

    @staticmethod
    def plot_nof_requests(matrix, taxi_pos, proposed_action, t):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hm = sns.heatmap(matrix, ax=ax, cmap='Greens', vmin=0, vmax=65, square=True, cbar_kws={"shrink": .75})
        hm.set_ylim((0, matrix.shape[0]))
        plt.xlabel('x-index of grid', fontname=font), plt.ylabel('y-index of grid', fontname=font)
        plt.yticks(fontname=font, rotation=0), plt.xticks(fontname=font)
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        cb = ax.collections[0].colorbar
        label_values = [0, 10, 20, 30, 40, 50, 60]
        cb.set_ticks([i for i in label_values])
        cb.set_ticklabels([str(i) for i in label_values], font=font)
        # plt.show()
        plt.savefig(f'./explanation/graphics/state_r/{t}', bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_r_error(matrix, taxi_pos, proposed_action, t):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hm = sns.heatmap(matrix, ax=ax, cmap='coolwarm', vmin=-15, vmax=15, center=0, square=True, cbar_kws={"shrink": .75})
        hm.set_ylim((0, matrix.shape[0]))
        plt.xlabel('x-index of grid', fontname=font), plt.ylabel('y-index of grid', fontname=font)
        plt.yticks(fontname=font, rotation=0), plt.xticks(fontname=font)
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        cbar = ax.collections[0].colorbar
        label_values = cbar.get_ticks()
        cbar.set_ticks([i for i in label_values])
        labels = [f'{i:.0f}' for i in label_values]
        labels[0], labels[-1] = f'$\leq$ {label_values[0]}', f'$\geq$ {label_values[-1]}'
        cbar.set_ticklabels(labels, font=font)
        # plt.show()
        plt.savefig(f'./explanation/graphics/state_r_hat_minus_r/{t}', bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_action_map(explanation, taxi_pos, proposed_action, file_name):
        mpl.rcParams['figure.dpi'] = 300
        min_diff, max_diff = np.asarray([a[4] for a in explanation]).min(), np.asarray([a[4] for a in explanation]).max()
        matrix = np.zeros((20, 20))
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hm = sns.heatmap(matrix, ax=ax, cmap='Greys', vmin=0, vmax=1000, square=True, cbar=False)
        hm.set_ylim((0, matrix.shape[0]))
        plt.xlabel('x-index of grid', fontname=font), plt.ylabel('y-index of grid', fontname=font)
        plt.yticks(fontname=font, rotation=0), plt.xticks(fontname=font)
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        cmap = plt.get_cmap('summer_r')
        for a in explanation:
            alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
            hm.arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], color=cmap(alpha), head_width=0.1)
        cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=plt.get_cmap('YlGn')), shrink=0.6)
        cb.outline.set_visible(False)
        cb.set_ticks([0, 1.0])
        cb.set_ticklabels(['0', '1'], font=font)
        cb.set_label('0 is uncertain and 1 certain', font=font)
        # plt.show()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()


    @staticmethod
    def plot_index(matrix, taxi_pos, proposed_action, file_name, reward_based=False):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hm = sns.heatmap(matrix, square=True, cmap='RdYlGn', ax=ax, cbar_kws={"shrink": .75})
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, .5, .75, 1.0])
        if reward_based:
            cbar.set_ticklabels(['-0.96', '0', '10', '20'], font=font)
        else:
            cbar.set_ticklabels(['0', '1', '2', '$\geq$3'], font=font)
        # plt.show()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()


    @staticmethod
    def plot_sarfa(matrix, taxi_pos, proposed_action, file_name):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        explanation_cmap = plt.get_cmap('Purples')
        # hm = sns.heatmap(matrix, square=True, cmap=explanation_cmap, ax=ax, cbar=False)
        hm = sns.heatmap(np.flip(matrix.T, axis=1), square=True, cmap=explanation_cmap, ax=ax, cbar=False, vmin=0, vmax=.25)
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        cb = hm.figure.colorbar(plt.cm.ScalarMappable(cmap=explanation_cmap), shrink=0.75)
        cb.outline.set_visible(False)
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(['0', '0.125', '0.25', '0.375', '0.5'], font=font)
        # cb.set_label('0 is no influence and 1 is a strong influence', font=font)
        # plt.show()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_shap(matrix, taxi_pos, proposed_action, vmin, vmax, file_name):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        explanation_cmap = plt.get_cmap('bwr')
        hm = sns.heatmap(matrix.T, square=True, cmap=explanation_cmap, ax=ax, cbar=True, center=0, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .75})
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        # plt.show()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_lime(matrix, taxi_pos, proposed_action, vmin, vmax, file_name):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        explanation_cmap = plt.get_cmap('bwr')
        hm = sns.heatmap(matrix, square=True, cmap=explanation_cmap, ax=ax, cbar=True, center=0, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .75})
        if taxi_pos is not None: hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        if proposed_action is not None: hm.add_patch(Rectangle(proposed_action, 1, 1, fill=False,
                                                               edgecolor='cornflowerblue', lw=3))
        ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)

        cbar = ax.collections[0].colorbar
        label_values = cbar.get_ticks()
        cbar.set_ticks([i for i in label_values])
        cbar.set_ticklabels([f'{i:.1f}' for i in label_values], font=font)

        plt.savefig(file_name, bbox_inches='tight')
        plt.close()