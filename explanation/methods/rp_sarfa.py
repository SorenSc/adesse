"""
The following is taken from https://arxiv.org/pdf/2101.07312.pdf and realizes the XAI method SARFA, which was proposed
by https://arxiv.org/abs/1912.12191.

The code is derived from the corresponding repository:
https://raw.githubusercontent.com/belimmer/PerturbationSaliencyEvaluation/main/applications/atari/sarfa.py

Date: 09-2022
commit: fa51a232f7356f249c6038480274e107e48aab63
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from scipy.stats import entropy

from datamodel.datamodel import RPState
from explanation.explainer import RepositioningAgentExplainer


class SarfaExplainer(RepositioningAgentExplainer):
    """
    For creating saliency maps using SARFA (https://arxiv.org/pdf/1912.12191.pdf)
    """

    def __init__(self):
        self.name = 'SARFA'

    occlude = lambda I, mask: I * (1 - mask)
    occlude_blur = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask  # choose an area to blur

    @staticmethod
    def cross_entropy(original_output, perturbed_output, action_index):
        # remove the chosen action in original output
        p = original_output[:action_index]
        p = np.append(p, original_output[action_index + 1:])
        # According to equation (2) in the paper(https://arxiv.org/abs/1912.12191v4)
        # the softmax should happen over the out put with the chosen action removed.
        # We do it like this here but we want to mention that this differs from the
        # implementation in https://github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py
        p = softmax(p)

        # Do the same for the perturbed output
        new_p = perturbed_output[:action_index]
        new_p = np.append(new_p, perturbed_output[action_index + 1:])
        new_p = softmax(new_p)

        # According to the paper this should be the other way around: entropy(new_p,p)
        # (directly und er equation (2) in https://arxiv.org/pdf/1912.12191.pdf)
        # While this would make a difference, it is like this in the official implementation in
        # github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py:
        KL = entropy(p, new_p)

        K = 1. / (1. + KL)

        return K

    def sarfa_saliency(self, original_output, perturbed_output, action_index):
        """
        Calculate the impact of the perturbed area in *perturbed_output* for the action *action_index*
        according to the SARFA formula.
        """
        original_output = np.squeeze(original_output.detach().numpy())
        perturbed_output = np.squeeze(perturbed_output.detach().numpy())
        dP = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
        if dP > 0:
            K = self.cross_entropy(original_output, perturbed_output, action_index)
            return (2 * K * dP) / (K + dP)
        else:
            return 0

    def get_occlusion_mask(self, center, size, radius):
        """
        Creates a mask to occlude the image with black color

        Args:
            center: center position of the mask
            size: size of the mask
            radius: the radius of the mask

        Returns:
            mask: The newly created mask

        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        # distance to center(calculated with pythagoras) has to be lower then or equal to radius
        keep = x * x + y * y <= radius * radius
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        return mask

    def get_pertubation_mask(self, y_i, x_i, z_i, size):
        """
        Creates a mask of the given size for the index - composed of `x_i` and `y_i` and `z_i`.
        :param x_i: x-index of the grid
        :param y_i: y-index of the grid
        :param z_i: z-index - refers to pick-up or drop-off
        :param size: size of the input passed into the model
        :return: returns a mask of the given size that masks the composed index
        """
        index = np.s_[y_i, x_i, z_i]
        mask = np.zeros(size)
        mask[index] = 1.0
        return mask

    def get_blur_mask(self, center, size, radius):
        """
        Creates the blurred mask, which will be added to the  image.

        Args:
            center: center position of mask
            size: size of the mask
            radius: radius of the blurring

        Returns:
            mask: mask which is used to perturb images
        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    @staticmethod
    def perturbate(input, mask, replacement=0.0):
        """
        Here, the given input is perturbed at indices marked in the mask; for the perturbation the given replacement
        value is used.
        :param input:
        :param mask:
        :param replacement:
        :return: Returns the perturbated input.
        """

        # Transform to torch mask
        dropoff_mask = torch.tensor(mask[:, :, 0], dtype=torch.bool)
        pickup_mask = torch.tensor(mask[:, :, 1], dtype=torch.bool)

        # Clone objects to only modify one value at a time
        dropoff_demand = torch.clone(input.dropoff_demand)
        predicted_pickup_demand = torch.clone(input.predicted_pickup_demand)

        # Apply replacement
        dropoff_demand[0, 0][dropoff_mask] = replacement
        predicted_pickup_demand[0, 0][pickup_mask] = replacement

        # Create new `RPState` with modified values
        return RPState(input.delta, dropoff_demand, predicted_pickup_demand, input.taxis)

    def explain(self, input, model, action, original_output):
        """
        Generates a SARFA explanation for the prediction of a given model.
        :param input: Input that is perturbed for explaining.
        :param model: Model that shall be explained.
        :param action: Action with highest Q-value.
        :param original_output: Model(input)
        :return: Returns a three-dimensional explanation for each of the given inputs - this is a saliency map.
        """

        x, y, z = 20, 20, 2
        saliency_map = np.zeros((x, y, z))

        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # In our case, we are not using a bunch of images from different time steps as input, but multiple
                    # 'images' with different content. Accordingly, we cannot simply occlude the same pixel in each z,
                    # but need to handle each feature in each dimension - also z - as a unique feature and occlude it
                    # accordingly.
                    # Using a mask is not necessarily required when only one value is changed at a time, but once
                    # superpixels are used, their usage makes more sense.
                    mask = self.get_pertubation_mask(y_i=j, x_i=i,  z_i=k, size=[x, y, z])
                    perturbed_input = self.perturbate(input, mask, 10)
                    perturbed_output = model((perturbed_input.predicted_pickup_demand, perturbed_input.dropoff_demand,
                                             perturbed_input.taxis[0].location))
                    saliency_map[j, i, k] = self.sarfa_saliency(original_output, perturbed_output, action)

        return saliency_map