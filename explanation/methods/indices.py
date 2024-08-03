import random
from abc import ABC, abstractmethod

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from config.commons import get_configs
from dataset import CityCenterDataset
from dataset import DatasetCommons


class StateIndex(ABC):

    @abstractmethod
    def compute(self, arr) -> float:
        """
        Computes formula that the index is based on.
        :param arr:
        :return:
        """
        pass

    @abstractmethod
    def visualize(self, matrix, sample=False):
        """
        Visualizes the given matrix.
        :param matrix:
        :return:
        """
        pass

    @abstractmethod
    def div_norm(self, v) -> float:
        """
        As colors.TwoSlopeNorm does not work properly with small values, this function returns for
        a given value the corresponding value for the colorscale. We need such function as our
        colorscales do not have the zero in the center.
        :param v:
        :return: A float value between 0.0 and 1.0. 0.5 refers to the center of the colorscale.
        """
        pass


class RequestTaxiIndex(StateIndex):

    def compute(self, arr) -> float:
        nof_requests, nof_taxis = arr[0], arr[1]
        return nof_requests / nof_taxis

    def visualize(self, matrix, sample=False):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        if sample:
            ax.set_title('Sample for Idea 1', font=font)
        else:
            ax.set_title('Idea 1: $\^r / t$', font=font)
        sns.heatmap(matrix, square=True, cmap='RdYlGn', ax=ax, cbar_kws={"shrink": .75})
        if sample:
            ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        else:
            ax.set_xlabel('$t$', fontname=font), ax.set_ylabel('$\^r$', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, .5, .75, 1.0])
        cbar.set_ticklabels(['0', '1', '2', '$\geq$3'], font=font)
        if sample:
            plt.savefig('./explanation/graphics/SampleIdea1.PNG', bbox_inches='tight')
        else:
            plt.savefig('./explanation/graphics/Idea1.PNG', bbox_inches='tight')
        plt.show()
        plt.close()

    def div_norm(self, v) -> float:
        if v < 1:
            return v / 1 * 0.5
        elif v > 1:
            return 0.5 + min(0.5, ((v - 1) / 2 * 0.5))
        else:
            return 0.5


class RewardBasedIndex(StateIndex):

    def compute(self, arr) -> float:
        nof_requests = arr[0]  # Can be an estimation
        nof_taxis = arr[1]
        delta = nof_requests - nof_taxis
        if delta >= 2:
            return 20
        elif delta == 1:
            return 10
        elif (delta <= 0) and nof_requests > 0:
            return 10 * (nof_requests / nof_taxis)
        else:
            return (24 * -1 + 0) / 25

    def visualize(self, matrix, sample=False):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        if sample:
            ax.set_title('Sample for Idea 2', font=font)
        else:
            ax.set_title('Idea 2: Reward-Based Index', font=font)
        sns.heatmap(matrix, square=True, cmap='RdYlGn', ax=ax, cbar_kws={"shrink": .75})
        if sample:
            ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        else:
            ax.set_xlabel('$t$', fontname=font), ax.set_ylabel('$\^r$', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, .5, .75, 1.0])
        cbar.set_ticklabels(['-0.96', '0', '10', '20'], font=font)
        if sample:
            plt.savefig('./explanation/graphics/SampleIdea2.PNG', bbox_inches='tight')
        else:
            plt.savefig('./explanation/graphics/Idea2.PNG', bbox_inches='tight')
        plt.show()
        plt.close()


    def div_norm(self, v) -> float:
        if v < 0:
            return (v + .96) / .96 * 0.5
        elif v > 0:
            return 0.5 + (v / 20 * 0.5)
        else:
            return 0.5


class WeightedRequestTaxiIndex(StateIndex):

    def __init__(self, alpha=.5, mean=None):
        self.alpha = alpha
        self.mean = mean

    def compute(self, arr) -> float:
        nof_requests, nof_taxis = arr[0], arr[1]
        return self.alpha * nof_requests / nof_taxis + (1 - self.alpha) * nof_requests / self.mean

    def visualize(self, matrix, sample=False):
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        if sample:
            ax.set_title('Sample for Idea 3', font=font)
        else:
            ax.set_title('Idea 3: Weighted Request-Taxi Index', font=font)
        sns.heatmap(matrix, square=True, cmap='RdYlGn', ax=ax, cbar_kws={"shrink": .75})
        if sample:
            ax.set_xlabel('x-index', fontname=font), ax.set_ylabel('y-index', fontname=font)
        else:
            ax.set_xlabel('$t$', fontname=font), ax.set_ylabel('$\^r$', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, .5, .75, 1.0])
        cbar.set_ticklabels(['0', '1', '2', '$\geq 3$'], font=font)
        if sample:
            plt.savefig('./explanation/graphics/SampleIdea3.PNG', bbox_inches='tight')
        else:
            plt.savefig('./explanation/graphics/Idea3.PNG', bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def div_norm(v) -> float:
        if v < 1.0:
            return .5 - v * .5
        elif v > 1.0:
            return .5 + min(.5, min(3, v) / 2 * .5)
        else:
            return .5


class StateIndexCommons:

    @classmethod
    def visualize_frequency_of_demand_supply_indeces(cls):
        pu_config, do_config = get_configs()
        random.seed(pu_config.random_seed)
        idx_all, _, _, _, idx_train_rp, idx_valid_rp, idx_test_rp = DatasetCommons.get_train_validation_test_indices(pu_config)
        pu_ds = CityCenterDataset(pu_config, idx_all)
        do_ds = CityCenterDataset(do_config, idx_all)

        max_value = 50
        result = np.zeros((max_value, max_value))
        for k in range(25):
            idx = random.choice(idx_all)
            _, pu_y = pu_ds.__getitem__(idx)
            _, do_y = do_ds.__getitem__(idx)
            for i in range(20):
                for j in range(20):
                    if (pu_y[i, j] < max_value) and (do_y[i, j] < max_value):
                       result[int(pu_y[i, j]), int(do_y[i, j])] += 1

        result[result == 0] = 0.0001
        matrix = result
        cmap = sns.color_palette("rocket", as_cmap=True)
        mpl.rcParams['figure.dpi'] = 300
        font, grid_size = 'Consolas', matrix.shape[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_title('Frequency of Cases')
        sns.heatmap(matrix, square=True, cmap=cmap, norm=LogNorm(), ax=ax, cbar_kws={"shrink": .75})
        ax.set_xlabel('$t$', fontname=font), ax.set_ylabel('$\^r$', fontname=font)
        ax.set_xlim(0, grid_size), ax.set_ylim(0, grid_size)
        plt.xticks(fontname=font, rotation=0), plt.yticks(fontname=font, rotation=0)
        for tick in ax.get_xticklabels(): tick.set_fontname(font)
        for tick in ax.get_yticklabels(): tick.set_fontname(font)
        plt.savefig('./explanation/graphics/IndexFrequencyOfCases.PNG', bbox_inches='tight')
        # plt.show()
        plt.close()


    @classmethod
    def compute_mean_of_the_predicted_nof_taxis(cls):

        pu_config, do_config = get_configs()
        random.seed(pu_config.random_seed)
        idx_all, _, _, _, idx_train_rp, idx_valid_rp, idx_test_rp = DatasetCommons.get_train_validation_test_indices(pu_config)
        pu_ds = CityCenterDataset(pu_config, idx_all)

        result = []
        for k in range(10000):
            idx = random.choice(idx_all)
            _, pu_y = pu_ds.__getitem__(idx)
            result.append(pu_y.mean().tolist())
        return np.asarray(result).mean()  # Sth like # 5.835649499675445




def main():
    grid_size = 30
    grid = np.array(np.meshgrid(range(grid_size), range(grid_size))).T.reshape(-1, 2)

    state_index = RequestTaxiIndex()
    state_index = WeightedRequestTaxiIndex()

    matrix = np.apply_along_axis(state_index.compute, axis=1, arr=grid).reshape(grid_size, grid_size)
    matrix = np.asarray([state_index.div_norm(e) for e in matrix.flatten()]).reshape(30, 30)
    state_index.visualize(matrix)

if __name__ == '__main__':
    main()
