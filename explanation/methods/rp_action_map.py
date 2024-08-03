import numpy as np
import torch

from environment.environment import EnvironmentCommons
from explanation.explainer import RepositioningAgentExplainer


class ActionMapExplainer(RepositioningAgentExplainer):
    """
    To create explanations via action maps or showing off which action is most promising from each potential taxi
    location.
    """

    def __init__(self):
        self.name = 'ActionMap'

    @staticmethod
    def action_dim_to_arrow_length(value):
        """
        Transform one dimension of an action to an arrow length that can be used in the visualization of the Q-values.
        :param value:
        :return:
        """
        if value == 0:
            return 0
        elif abs(value) == 1:
            return np.sign(value) * .325
        elif abs(value) == 2:
            return np.sign(value) * .65

    def explain(self, simulator, comparison='ToMin'):
        """
        Iterate through all potential positions, select the most promising action from this
        position, and store the
        corresponding arrow.
        :param net:
        :param simulator:
        :return:
        """
        result = np.zeros((20, 20, 4))
        for pos_x in range(20):
            for pos_y in range(20):
                taxi_pos = torch.tensor([[pos_x, pos_y]])

                # Get the Q-values for the current state and given taxi position.
                q_values = simulator.policy_net((simulator.env.state.predicted_pickup_demand,
                                  simulator.env.state.dropoff_demand, taxi_pos))

                if comparison == 'ToMin':
                    min_q_value = q_values.min().detach().item()
                elif comparison == 'ToSecondHighest':
                    min_q_value = q_values.topk(2).values[0][1].detach().item()
                else:
                    print(f'Comparison method {comparison} does not exist.')
                    break
                max_q_value = q_values.max().detach().item()

                # Filter out impossible actions by setting their Q-value lower
                action_filter = simulator.create_filter_for_valid_actions(taxi_pos)
                q_values[0][~action_filter] = -5

                # Select action with highest Q-value or exploit the policy
                action = q_values.max(1)[1].view(1, 1)

                # Create arrow
                two_dim_action = EnvironmentCommons.single_to_two_dimensional_action(action)
                arrow_dx = self.action_dim_to_arrow_length(two_dim_action[0][0].item())
                arrow_dy = self.action_dim_to_arrow_length(two_dim_action[0][1].item())
                result[pos_x, pos_y] = arrow_dx, arrow_dy, min_q_value, max_q_value
