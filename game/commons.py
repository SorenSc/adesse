import json

import numpy as np
import torch
import seaborn as sns
import streamlit.components.v1 as components
from environment.environment import EnvironmentCommons

from explanation.methods.no_explanation import NoExplanation
from explanation.methods.rp_action_map import ActionMapExplainer
from explanation.methods.rp_sarfa import SarfaExplainer

   
def get_repositioning_plots(state_included, plot_titles, predicted_pickup_demand, dropoff_demand, rp_exp_method,
                            rp_explanation, taxi_location, proposed_action, next_locations):
    """
    Creates the predicted pick-up demand and the drop-off demand plot.
    :param next_locations:
    :param taxi_location:
    :param rp_explanation:
    :param dropoff_demand:
    :param predicted_pickup_demand:
    :param rp_exp_method:
    :param proposed_action:
    :return:
    """
    explainer, explanation = NoExplanation(), None  # 'NoExplanation' case
    if rp_exp_method == 'ActionMap':
        explanation = []
        for e in rp_explanation:
            explanation.append(((e[0], e[1]), (e[2], e[3]), e[4], e[5], e[6]))
        explainer = ActionMapExplainer()
    elif rp_exp_method == 'SARFA':
        explainer = SarfaExplainer()
        explanation = np.asarray(json.loads(rp_explanation))
    elif rp_exp_method == 'SHAP':
        explainer = NoExplanation()
        explainer.name = 'SHAP'
        explanation = np.asarray(json.loads(rp_explanation))
    return explainer.visualize_explanation(
        state_included, plot_titles, taxi_location, predicted_pickup_demand, dropoff_demand, explanation, 0,
        proposed_action, next_locations, True)


def get_impossible_actions_as_single_str(action_filter) -> str:
    """
    Based on a given action filter, all corresponding/impossible action buttons are identified, and their labels
    are put in a list, which is then joined to a single string and returned. It needs to be a single string as we
    later modify JS from Python and passing the list as list or alike is not possible with Streamlit.
    :param action_filter:
    """
    impossible_actions_1d = torch.arange(0, 25, dtype=int)[~action_filter].tolist()
    impossible_actions = [EnvironmentCommons.single_to_two_dimensional_action(a).tolist()[0] for a in
                          impossible_actions_1d]
    return '; '.join([f'{a[0]}, {a[1]}' for a in impossible_actions])


def mark_action_buttons(proposed_action, taxi_location):
    """
    This function marks the taxi location and the proposed action in the action buttons. It is realized in a pretty
    hacky way, but Streamlit provides so far no other option to modify single buttons.
    :param proposed_action:
    :param taxi_location:
    :return:
    """
    active_btn = f'{proposed_action[0] - taxi_location[0]}, ' \
                 f'{proposed_action[1] - taxi_location[1]}'
    components.html(f'''<script>
    const action_btn_labels = '-2, 2; -1, 2; 0, 2; 1, 2; 2, 2; -2, 1; -1, 1; 0, 1; 1, 1; 2, 1; -2, 0; -1, 0; 0, 0; 1, 0; 2, 0; -2, -1; -1, -1; 0, -1; 1, -1; 2, -1; -2, -2; -1, -2; 0, -2; 1, -2; 2, -2';
    const btns = window.parent.document.querySelectorAll('.stButton > button')
    btns.forEach(btn => {{
        if(action_btn_labels.includes(btn.textContent))
            if(btn.textContent == '{active_btn}')
                btn.style.backgroundColor = '#B9CCF1FF'
            else
                (btn.textContent == '0, 0') ? (btn.style.backgroundColor = '#F3D87FFF') : (btn.style.backgroundColor = '#ffffff')
    }})
    </script>''', height=0, width=0)


def disable_impossible_actions(impossible_actions):
    """
    Disables all action buttons that are impossible; some actions are impossible as the taxi is in the corner of the
    city.
    :param impossible_actions:
    :return:
    """
    components.html(f'''<script>
    const action_btn_labels = '-2, 2; -1, 2; 0, 2; 1, 2; 2, 2; -2, 1; -1, 1; 0, 1; 1, 1; 2, 1; -2, 0; -1, 0; 0, 0; 1, 0; 2, 0; -2, -1; -1, -1; 0, -1; 1, -1; 2, -1; -2, -2; -1, -2; 0, -2; 1, -2; 2, -2';
    const impossible_actions = '{impossible_actions}'.split('; ');
    const btns = window.parent.document.querySelectorAll('.stButton > button');
    btns.forEach(btn => {{
        if(action_btn_labels.includes(btn.textContent))
            btn.disabled = false;
    }})
    btns.forEach(btn => {{
        if(action_btn_labels.includes(btn.textContent))
            if(impossible_actions.includes(btn.textContent))
                btn.disabled = true
    }})
    </script>''', height=0, width=0)


def mark_dp_explanation_buttons(btn_label):
    """
    Marks the button for the currently selected demand prediction explanation.
    :param btn_label:
    :return:
    """
    components.html(f'''<script>
    const dp_exp_btns_strs = 'Taxi; Recommended; A; B; C; D'.split('; ');
    const btns = window.parent.document.querySelectorAll('.stButton > button')
    btns.forEach(btn => {{
        if(dp_exp_btns_strs.includes(btn.textContent))
            if(btn.textContent == '{btn_label}')
                btn.style.backgroundColor = '#A7F1F1'
            else
                btn.style.backgroundColor = '#ffffff'
    }})
    </script>''', height=0, width=0)


def disable_impossible_dp_exp_buttons(impossible_buttons):
    """
    Some demand prediction explanations are not possible as the next steps repeat each other, so that, we do not reach
    up to location D. Consequently, impossible buttons are made inaccessible.
    :param impossible_buttons:
    :return:
    """
    components.html(f'''<script>
        const dp_exp_btns_strs = 'Taxi; Recommended; A; B; C; D'.split('; ');
        const impossible_btns = '{impossible_buttons}'.split('; ');
        const btns = window.parent.document.querySelectorAll('.stButton > button')
        btns.forEach(btn => {{
            if(dp_exp_btns_strs.includes(btn.textContent))
                btn.disabled = false;
        }})
        btns.forEach(btn => {{
            if(dp_exp_btns_strs.includes(btn.textContent))
                if(impossible_btns.includes(btn.textContent))
                    btn.disabled = true
        }})
    </script>''', height=0, width=0)


def visualize_plots(vis_elements, figure_ppu, figure_dp, state_included, rp_exp_method):
    """
    Visualizes the two given plots in the Streamlit game.
    :param vis_elements:
    :param figure_ppu:
    :param figure_dp:
    :return:
    """
    if vis_elements['ppu_graph'] is not None: vis_elements['ppu_graph'].empty()
    vis_elements['ppu_graph'] = vis_elements['ppu_graph'].pyplot(figure_ppu)
    if vis_elements['do_graph'] is not None: vis_elements['do_graph'].empty()
    if state_included or (not state_included and rp_exp_method in ['SARFA', 'SHAP']):
        vis_elements['do_graph'] = vis_elements['do_graph'].pyplot(figure_dp)


def visualize_status(vis_elements, taxi_location, last_reward, accumulated_reward):
    """
    Visualizes the status - shown at the top of the game - which consists of the current taxi location, the last reward,
    and the accumulated reward.
    :param vis_elements:
    :param taxi_location:
    :param last_reward:
    :param accumulated_reward:
    :return:
    """
    vis_elements['location'].markdown(f'**[{taxi_location[0]}, {taxi_location[1]}]**')
    vis_elements['last_reward'].markdown(f'**{last_reward}**')
    vis_elements['acc_reward'].markdown(f'**{accumulated_reward}**')


def table_styler(styler):
    """
    Styles an HTML-table derived from a Pandas dataframe.
    :param styler:
    :return:
    """
    styler.background_gradient(cmap='coolwarm', subset=['Importance [in trips]'], vmin=-25, vmax=25)
    styler.format('{:.2f}', subset=['Importance [in trips]'])
    return styler





