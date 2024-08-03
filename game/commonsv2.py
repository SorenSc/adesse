import logging
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import numpy as np
import seaborn as sns
import streamlit.components.v1 as components
import torch


def visualize_baseline_explanation(matrix, last_pos, taxi_pos, action_2d, i, cbar=False, type='lime', file_name='sth.PNG'):

    if type == 'lime':
        vmin, vmax, cmap = -250, 250, 'PiYG'
    elif type == 'state':
        vmin, vmax, cmap = 0, 125, 'Greens'
    else:
        raise NotImplementedError

    mpl.rcParams['figure.dpi'] = 300
    font = 'Consolas'
    shrink_factor = .5
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    titles = ['#requests 30 minutes ago', '#requests 20 minutes ago', '#requests 10 minutes ago',
        '#requests now', '#taxis in 10 minutes']

    hm = sns.heatmap(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
        square=True, cbar=cbar, cbar_kws=dict(shrink=.75))
    hm.set_ylim((0, 20))

    ax.set_xlabel('x-index of grid', fontname=font)
    ax.set_ylabel('y-index of grid', fontname=font)
    # ax.set_title(titles[i], fontname=font)
    plt.xticks(fontname=font), plt.yticks(fontname=font, rotation=0)

    if last_pos is not None:
        hm.add_patch(Rectangle(last_pos, 1, 1, fill=False, edgecolor='black', lw=1))

    if action_2d is not None:
        hm.add_patch(Rectangle(action_2d, 1, 1, fill=False, edgecolor='cornflowerblue', lw=2))
        hm.annotate('B', (action_2d[0] + 0.45, action_2d[1] + 0.45), color='cornflowerblue', fontsize=12,
            ha='center', va='center', font=font, weight='bold')

    if taxi_pos is not None:
        hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        hm.annotate('A', (taxi_pos[0] + 0.45, taxi_pos[1] + 0.45), color='gold', fontsize=12,
            ha='center', va='center', font=font, weight='bold')

    if cbar and (type == 'lime'):
        cbar_index = ax.collections[0].colorbar
        cbar_index.set_ticks([vmin, 0, vmax])
        cbar_index.set_ticklabels(['Negative', '0', 'Positive'], font=font)
        cbar_index.set_label('Influence', labelpad=-25, font=font)
    elif cbar and (type == 'state'):
        cbar_index = ax.collections[0].colorbar
        cbar_index.set_ticks([0, 25, 50, 75, 100, 125])
        cbar_index.set_ticklabels(['0', '25', '50', '75', '100', '>=125'], font=font)
        cbar_index.set_label('#taxis or #requests', font=font)

    plt.savefig(f'./logs/graphics/{file_name}', bbox_inches='tight')
    plt.close()
    return fig


def visualize_compositional_explanation(explanation, last_pos, taxi_pos, advice, game=False, file_name='sth.png'):
    """
    Visualize the arrows - together with the uncertainty - on top of the index.
    """
    mpl.rcParams['figure.dpi'] = 300
    font = 'Consolas'
    shrink_factor = .5
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot index
    matrix = np.asarray(explanation['index'])
    matrix = np.asarray([div_norm(e) for e in np.nditer(matrix)]).reshape(20, 20)
    hm = sns.heatmap(matrix, ax=ax, cmap='RdYlGn', square=True, vmin=0, vmax=1,
        cbar_kws=dict(shrink=shrink_factor, location='left'), alpha=.5)

    # Plot arrows
    colors = [(0.8501191849288735, 0.8501191849288735, 0.8501191849288735), (0.0, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list('GreyToBlack', colors, N=100)
    arrows = np.asarray(explanation['arrows'])
    delta = np.asarray(arrows[:, :, 3] - arrows[:, :, 2])
    min_delta, max_delta = delta.min(), delta.max()
    for x_i, x in enumerate(arrows):
        for y_i, y in enumerate(arrows):
            alpha = (delta[x_i, y_i] - min_delta) / (max_delta - min_delta)
            hm.arrow(x_i + .5, y_i + .5, arrows[x_i, y_i, 0], arrows[x_i, y_i, 1],
                head_width=0.1, color=cmap(alpha))

    # if last_pos is not None:
    #     hm.add_patch(Rectangle(last_pos, 1, 1, fill=False, edgecolor='black', lw=1))

    if advice is not None:
        hm.add_patch(Rectangle(advice, 1, 1, fill=False, edgecolor='cornflowerblue', lw=2))
        hm.annotate('B', (advice[0] + 0.45, advice[1] + 0.45), color='royalblue', fontsize=15,
            ha='center', va='center', font=font, weight='bold')

    if taxi_pos is not None:
        hm.add_patch(Rectangle(taxi_pos, 1, 1, fill=False, edgecolor='gold', lw=2))
        hm.annotate('A', (taxi_pos[0] + 0.45, taxi_pos[1] + 0.45), color='gold', fontsize=15,
            ha='center', va='center', font=font, weight='bold')
        # Usage of taxi logo - the taxi logo is not used as the index value wouldn't be visible
        # taxi_logo = image.imread('./game/data/taxi.png')
        # imagebox = OffsetImage(taxi_logo, zoom=0.025)
        # ab = AnnotationBbox(imagebox, (taxi_pos[1]+.5, taxi_pos[0]+.5), frameon=False)
        # ax.add_artist(ab)

    for i, l in enumerate(explanation['locations']):
        if i > 1:
            hm.add_patch(Rectangle(l, 1, 1, fill=False, edgecolor='mediumorchid', lw=2))
            hm.annotate(chr(ord('C') + i - 2), (l[0] + 0.45, l[1] + 0.45), color='mediumorchid', 
                fontsize=15, ha='center', va='center', font=font, weight='bold')

    hm.set_ylim((0, 20))
    plt.xlabel('x-index of grid', fontname=font), plt.ylabel('y-index of grid', fontname=font)
    plt.yticks(fontname=font, rotation=0), plt.xticks(fontname=font)

    cbar_index = ax.collections[0].colorbar
    cbar_index.set_ticks([0, 1.0])
    cbar_index.set_ticklabels(['Bad', 'Good'], font=font)
    cbar_index.set_label('Taxi Index', labelpad=-25, font=font)

    cbar_arrows = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, shrink=shrink_factor)
    cbar_arrows.outline.set_visible(False)
    cbar_arrows.set_ticks([0, 1.0])
    cbar_arrows.set_ticklabels(['0', '1'], font=font)
    cbar_arrows.set_label('Importance', labelpad=-10, font=font)  # -50

    plt.savefig(f'./logs/graphics/{file_name}', bbox_inches='tight')
    plt.close()
    return fig


def div_norm(v) -> float:
    """
    Used to transfer values of the index to a color. 
    """
    if v < 1.0:
        return .5 - (1.0 - v) * .5
    elif v > 1.0:
        return .5 + min(.5, min(3, v) / 2 * .5)
    else:
        return .5


def single_to_two_dimensional_action(action: torch.Tensor):
    """
    Transforms a single dimensional action into a two-dimensional one. For instance, the action with number two
    refers to (0, 2) or an movement towards the top of the environment. The first value represents the action on the
    x-axis and the second value the action on the y-axis.
    :param device:
    :param action: Something like tensor([[22]]); value has to be between 0 and 24.
    :return: (movement in x-direction, movement in y-direction)
    """
    return torch.tensor([[action % 5 - 2, (torch.div(action, 5, rounding_mode='trunc') - 2) * -1]])


def get_impossible_actions_as_single_str(action_filter) -> str:
    """
    Based on a given action filter, all corresponding/impossible action buttons are identified, and their labels
    are put in a list, which is then joined to a single string and returned. It needs to be a single string as we
    later modify JS from Python and passing the list as list or alike is not possible with Streamlit.
    :param action_filter:
    """
    impossible_actions_1d = torch.arange(0, 25, dtype=int)[~action_filter].tolist()
    impossible_actions = [single_to_two_dimensional_action(a).tolist()[0] for a in
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
    const dp_exp_btns_strs = 'A/Taxi; B/Advice; C; D; E; F'.split('; ');
    const btns = window.parent.document.querySelectorAll('.stButton > button')
    btns.forEach(btn => {{
        if(dp_exp_btns_strs.includes(btn.textContent))
            if(btn.textContent == '{btn_label}')
                btn.style.backgroundColor = '#d8b2eb'
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
        const dp_exp_btns_strs = 'A/Taxi; B/Recommended; C; D; E; F''.split('; ');
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


def table_styler(styler):
    """
    Styles an HTML-table derived from a Pandas dataframe.
    :param styler:
    :return:
    """
    styler.background_gradient(cmap='coolwarm', subset=['Importance'], vmin=-1.5, vmax=1.5)
    styler.format('{:.2f}', subset=['Importance'])
    return styler
