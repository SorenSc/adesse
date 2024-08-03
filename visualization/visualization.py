import os

import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib as mpl

from environment.environment import RepositioningEnvironment

mpl.rcParams['figure.dpi'] = 300
font = 'Consolas'


def make_state_transparent(hm_data, mm, line_plot_data_v2, repositionings_via_drop_off, repositionings_via_agent, t, title):
    pu_cmap = 'Greens'
    do_cmap = 'PuRd'
    delta_cmap = sns.color_palette('bwr', as_cmap=True)
    q_cmap = sns.color_palette('coolwarm_r', as_cmap=True)
    legend_font = font_manager.FontProperties(family=font, style='normal', size=9)

    fig, ax = plt.subplots(2, 4, figsize=(18, 7))  # (#plots on height, #plots on width), (width, height)
    fig.suptitle(hm_data['actual_time_readable'], fontname=font, fontsize=14)

    hm = sns.heatmap(hm_data['pu_y_p'], cmap=pu_cmap, vmin=0, vmax=mm[1], ax=ax[0, 0], cbar=True, square=True, cbar_kws={"shrink": .9})
    hm.add_patch(Rectangle(hm_data['new_taxi_pos'], 1, 1, fill=False, edgecolor='black', lw=2))
    hm.add_patch(Rectangle(hm_data['taxi_position'], 1, 1, fill=False, edgecolor='gold', lw=1))
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax[0, 0].set_title('Predicted Pick-Up Demand', fontname=font)

    hm = sns.heatmap(hm_data['do_y'], cmap=do_cmap, vmin=0, vmax=mm[1], ax=ax[0, 1], cbar=True, square=True, cbar_kws={"shrink": .9})
    hm.add_patch(Rectangle(hm_data['new_taxi_pos'], 1, 1, fill=False, edgecolor='black', lw=2))
    hm.add_patch(Rectangle(hm_data['taxi_position'], 1, 1, fill=False, edgecolor='gold', lw=1))
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax[0, 1].set_title('Drop-Off Demand', fontname=font)

    hm = sns.heatmap(hm_data['predicted_delta'], cmap=delta_cmap, center=0, vmin=mm[0], vmax=mm[1], ax=ax[0, 2], cbar=True, square=True, cbar_kws={"shrink": .9})
    hm.add_patch(Rectangle(hm_data['new_taxi_pos'], 1, 1, fill=False, edgecolor='black', lw=2))
    hm.add_patch(Rectangle(hm_data['taxi_position'], 1, 1, fill=False, edgecolor='gold', lw=1))
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    mae_delta = abs((hm_data['actual_delta'] - hm_data['predicted_delta'])).sum()/400
    ax[0, 2].text(0.65, 0.05, f'MAE: {mae_delta}', fontname=font, transform=ax[0, 2].transAxes)
    ax[0, 2].set_title('Delta', fontname=font)

    x_indices = line_plot_data_v2.keys()
    sns.lineplot(x=x_indices, y=[v[0] for v in line_plot_data_v2.values()], ax=ax[0, 3], color='royalblue', legend=False)
    sns.lineplot(x=x_indices, y=[v[1] for v in line_plot_data_v2.values()], ax=ax[0, 3], color='mediumseagreen', legend=False)
    sns.lineplot(x=x_indices, y=[v[2] for v in line_plot_data_v2.values()], ax=ax[0, 3], color='deeppink', legend=False)
    legend = ax[0, 3].legend(['#pick-ups', 'predicted #pick-ups', '#drop-offs'], frameon=True, prop=legend_font, loc='upper right')  # (0.025, 0.9))
    legend.get_frame().set_linewidth(0.0)
    ax[0, 3].axvline(hm_data['t'], color='black', alpha=.25)
    ax[0, 3].set_title('#Trips over Time', fontname=font)

    hm = sns.heatmap(hm_data['q-values'], cmap=q_cmap, ax=ax[1, 0], vmin=hm_data['q-values'].min(), vmax=hm_data['q-values'].max(), cbar=True, square=True, cbar_kws={"shrink": .9})
    hm.add_patch(Rectangle((2 + hm_data['action'][0], 2 + hm_data['action'][1] * -1), 1, 1, fill=False, edgecolor='black', lw=2))
    hm.add_patch(Rectangle((2, 2), 1, 1, fill=False, edgecolor='gold', lw=1))
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    ax[1, 0].set_title('Q-values', fontname=font)
    ax[1, 0].tick_params(bottom=False, left=False)
    ax[1, 0].set_yticklabels([]), ax[1, 0].set_xticklabels([])

    sns.lineplot(x=x_indices, y=[v[5] for v in line_plot_data_v2.values()], ax=ax[1, 1], color='black', legend=False)
    legend.get_frame().set_linewidth(0.0)
    ax[1, 1].axvline(hm_data['t'], color='black', alpha=.25)
    ax[1, 1].text(0.5, 0.05, f'Last reward: {hm_data["reward"]}', fontname=font, transform=ax[1, 1].transAxes)
    ax[1, 1].set_title('Acc. Reward over Time', fontname=font)

    ax[1, 2].scatter([line_plot_data_v2[t][3][0] + line_plot_data_v2[t][4][0]],
                     [line_plot_data_v2[t][3][1] + line_plot_data_v2[t][4][1]], color='black', marker='s')  # Action
    for key, a in repositionings_via_agent.items():
        ax[1, 2].arrow(a[0][0], a[0][1], a[1][0], a[1][1], head_width=0.1, color='black')
    for key, a in repositionings_via_drop_off.items():
        ax[1, 2].arrow(a[0][0], a[0][1], a[1][0], a[1][1], head_width=0.1, color='royalblue')
    ax[1, 2].scatter([line_plot_data_v2[t][3][0]], [line_plot_data_v2[t][3][1]], color='gold', marker='s')  # Taxi position
    ax[1, 2].set_xlim(0, 20), ax[1, 2].set_ylim(0, 20)
    ticks, tick_labels = list(range(0, 20, 2)), [str(e) for e in list(range(0, 20, 2))]
    ax[1, 2].set_xticks(ticks, font=font), ax[1, 2].set_xticklabels(ticks, font=font)
    ax[1, 2].set_yticks(ticks, font=font), ax[1, 2].set_yticklabels(ticks, font=font)
    ax[1, 2].set_title('Taxi Path', fontname=font)
    legend_elements = [Line2D([0], [0], color='black', lw=2, label='via agent'),
                       Line2D([0], [0], color='royalblue', lw=2, label='via drop-off')]
    ax[1, 2].legend(handles=legend_elements, loc='lower right', prop=legend_font, frameon=False)

    ax[1, 3].set_visible(False)

    for y in range(2):
        for x in range(4):
            for tick in ax[y, x].get_xticklabels(): tick.set_fontname(font)
            for tick in ax[y, x].get_yticklabels(): tick.set_fontname(font)

    plt.savefig(f'{title}.PNG', bbox_inches='tight')
    # plt.show()
    plt.close()


def visualize_training_effect_via_boxplots(config, result, title, step):

    mpl.rcParams['figure.dpi'] = 300
    nof_epi_to_visualize = config.n_episodes // step + 1
    fig, ax_arr = plt.subplots(1, 2, figsize=(28, 10))

    ax_arr[0].boxplot(result[0, :, :].T, medianprops=dict(color='royalblue'))
    ax_arr[0].title.set_text('Reward')
    ax_arr[1].boxplot(result[1, :, :].T, medianprops=dict(color='royalblue'))
    ax_arr[1].title.set_text('#pickups')

    for ax in ax_arr:
        ax.set_xticklabels([str(i * step) for i in range(nof_epi_to_visualize)])
        [tick.set_fontname(font) for tick in ax.get_xticklabels()]
        [tick.set_fontname(font) for tick in ax.get_yticklabels()]
        ax.set_xlabel('Episode', fontname=font)

    # plt.savefig(f'./results/{title}.PNG', bbox_inches='tight')
    plt.show()
    plt.close()


def visualize_training_effect(config, locations, rewards, v_pr_nr):
    """

    :param config:
    :param locations: (3, 2, 335, 2) - ((epochs 0, 10, 20, ... for visualization), n_runs, 7*48-1, (x, y)
    :param rewards:
    :param v_pr_nr:
    :return:
    """

    nof_epi_to_visualize = config.n_episodes // 10 + 1
    font = 'Consolas'
    fig, ax_arr = plt.subplots(4, nof_epi_to_visualize, figsize=(nof_epi_to_visualize * 5, 4 * 5))

    # Sample runs after training
    colors = ['royalblue', 'orange', 'mediumvioletred']
    for i in range(nof_epi_to_visualize):
        for j, letter in enumerate(['A', 'B', 'C'][:config.n_runs]):  # Limit visualization to first three paths

            # Plot path
            ax_arr[0, i].plot(locations[i][j][:, 1], locations[i][j][:, 0], lw=1, color=colors[j])
            ax_arr[0, i].scatter(locations[i][j][:, 1], locations[i][j][:, 0], s=2.5, color=colors[j])

            # Mark start point
            t = ax_arr[0, i].text(locations[i][j][0, 1], locations[i][j][0, 0], f'{letter} - Start', {'fontname': font})
            t.set_bbox({'facecolor': colors[j], 'alpha': 0.75, 'edgecolor': colors[j]})

            # Mark end point
            t = ax_arr[0, i].text(locations[i][j][-1, 1], locations[i][j][-1, 0], f'{letter} - End', {'fontname': font})
            t.set_bbox({'facecolor': colors[j], 'alpha': 0.75, 'edgecolor': colors[j]})

        # Kind of title
        t = ax_arr[0, i].text(0.5, 0.95, f'Sample Paths after {i * 10} Episodes', fontname=font, weight='bold', ha='center', va='center',
                              transform=ax_arr[0, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.25, 'edgecolor': 'white'})

        # Average reward
        t = ax_arr[0, i].text(0.5, 0.05, f'Average reward {rewards[i]:,.2f}', fontname=font, weight='bold', ha='center', va='center',
                              transform=ax_arr[0, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.25, 'edgecolor': 'white'})

        # Modify axes
        ax_arr[0, i].set_xlim(0, 20), ax_arr[0, i].set_ylim(0, 20)
        ax_arr[0, i].set_xticks(list(range(0, 20))), ax_arr[0, i].set_yticks(list(range(0, 20)))
        ax_arr[0, i].set_xticklabels([str(v) for v in list(range(0, 20))])
        ax_arr[0, i].set_yticklabels([str(v) for v in list(range(0, 20))])

    # Visualize #visits as well as the collected negative and positive reward per grid cell
    for i in range(nof_epi_to_visualize):
        sns.heatmap(v_pr_nr[0, i], cmap='plasma', vmin=0, ax=ax_arr[1, i], cbar=True, cbar_kws={"shrink": .9})
        sns.heatmap(v_pr_nr[1, i], cmap='coolwarm', vmin=0, center=0, ax=ax_arr[2, i], cbar=True,
                    cbar_kws={"shrink": .9})
        sns.heatmap(v_pr_nr[2, i], cmap='coolwarm', vmax=0, center=0, ax=ax_arr[3, i], cbar=True,
                    cbar_kws={"shrink": .9})

        # Titles
        t = ax_arr[1, i].text(0.5, 0.95, f'#Visits after {i * 10} Episodes', fontname=font, weight='bold',
                              ha='center', va='center', transform=ax_arr[1, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'white'})
        t = ax_arr[2, i].text(0.5, 0.95, f'Positive Reward after {i * 10} Episodes', fontname=font, weight='bold',
                              ha='center', va='center', transform=ax_arr[2, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'white'})
        t = ax_arr[3, i].text(0.5, 0.95, f'Negative Reward after {i * 10} Episodes', fontname=font, weight='bold',
                              ha='center', va='center', transform=ax_arr[3, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'white'})

        # Sums
        t = ax_arr[2, i].text(0.15, 0.05, f'Sum {int(v_pr_nr[1, i].sum())}', fontname=font,
                              ha='center', va='center', transform=ax_arr[2, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.25, 'edgecolor': 'white'})
        t = ax_arr[3, i].text(0.15, 0.05, f'Sum {int(v_pr_nr[2, i].sum())}', fontname=font,
                              ha='center', va='center', transform=ax_arr[3, i].transAxes)
        t.set_bbox({'facecolor': 'white', 'alpha': 0.25, 'edgecolor': 'white'})

        for j in [1, 2, 3]:
            ax_arr[j, i].set_xlim(0, 20), ax_arr[j, i].set_ylim(0, 20)
            ax_arr[j, i].set_xticks(list(range(0, 20, 2))), ax_arr[j, i].set_yticks(list(range(0, 20, 2)))
            ax_arr[j, i].set_xticklabels([str(v) for v in list(range(0, 20, 2))])
            ax_arr[j, i].set_yticklabels([str(v) for v in list(range(0, 20, 2))])
            plt.setp(ax_arr[j, i].get_yticklabels(), rotation=0)

    for j in range(4):
        for i in range(nof_epi_to_visualize):
            [tick.set_fontname(font) for tick in ax_arr[j, i].get_yticklabels()]
            [tick.set_fontname(font) for tick in ax_arr[j, i].get_xticklabels()]
    for i in range(1, nof_epi_to_visualize): ax_arr[1, i].axis('off')

    plt.savefig(f'./results/{config.model_name}', bbox_inches='tight')
    # plt.show()  # Too hard for memory ...
    plt.close(fig)


def simple_heatmap(matrix, title='', cmap='coolwarm'):
    sns.heatmap(matrix, cmap=cmap, vmin=matrix.min(), vmax=matrix.max(), center=0)
    plt.title(title, fontname=font)
    plt.xlabel('x-index of grid', fontname=font)
    plt.ylabel('y-index of grid', fontname=font)
    plt.yticks(fontname=font), plt.xticks(fontname=font)
    plt.xlim(0, matrix.shape[1])
    plt.ylim(0, matrix.shape[0])
    # plt.savefig(f'./visualization/graphics/{title}.PNG')
    plt.show()
    plt.close()


def simple_heatmap_with_locations(matrix, taxi_location, action_location=None, i=0, min=0, max=10, title='', cmap='coolwarm'):
    mpl.rcParams['figure.dpi'] = 300
    fig, ax = plt.subplots()
    sns.heatmap(matrix, cmap=cmap, vmin=min, vmax=max, center=0, ax=ax)
    plt.title(title, fontname=font)
    plt.xlabel('x-index of grid', fontname=font)
    plt.ylabel('y-index of grid', fontname=font)
    plt.yticks(fontname=font), plt.xticks(fontname=font)
    plt.xlim(0, matrix.shape[1])
    plt.ylim(0, matrix.shape[0])
    ax.add_patch(Rectangle((taxi_location[0]-2, taxi_location[1]-2), 5, 5, fill=False, edgecolor='royalblue', lw=1))
    ax.add_patch(Rectangle(taxi_location, 1, 1, fill=False, edgecolor='gold', lw=2))
    if action_location is not None:
        ax.add_patch(Rectangle(action_location, 1, 1, fill=False, edgecolor='black', lw=3))
    fig.savefig(f'./graphics/{title}_{str(i).rjust(2, "0")}.PNG', bbox_inches='tight')
    plt.close()


def visualize_transition(transition, cmap='coolwarm'):
    delta = transition.state.delta[0, 0].numpy()
    location = (transition.state.taxis[0].location.numpy()[0, 0],
                transition.state.taxis[0].location.numpy()[0, 1])
    next_location = (transition.next_state.taxis[0].location.numpy()[0, 0],
                     transition.next_state.taxis[0].location.numpy()[0, 1])
    reward_str = str(transition[3].numpy()[0])
    movement = RepositioningEnvironment.single_to_two_dimensional_action(transition.action)
    movement_str = '(' + str(movement[0][0,0].numpy()) + ', ' + str(movement[1][0,0].numpy()) + ')'
    title = f'Reward: {reward_str}; action: {movement_str}'

    ax = sns.heatmap(delta, cmap=cmap, vmin=delta.min(), vmax=delta.max(), center=0)

    ax.add_patch(Rectangle(location, 1, 1, fill=False, edgecolor='black', lw=4))
    ax.add_patch(Rectangle(next_location, 1, 1, fill=False, edgecolor='gold', lw=2))

    plt.title(title, fontname=font)
    plt.xlabel('x-index of grid', fontname=font)
    plt.ylabel('y-index of grid', fontname=font)
    plt.yticks(fontname=font), plt.xticks(fontname=font)
    plt.xlim(0, delta.shape[1])
    plt.ylim(0, delta.shape[0])

    plt.show()
    plt.close()


def visualize_pickup_demand_over_time(pu, do, delta, pu_y_at_t, do_y_at_t, delta_at_t, point_in_time, title):

    mpl.rcParams['figure.dpi'] = 300
    csfont = {'fontname': font}
    sns.set(style="white", rc={"lines.linewidth": 1})

    fig, axd = plt.subplot_mosaic([['left', 'center', 'right'], ['bottom', 'bottom', 'bottom']], figsize=(16, 8), dpi=300)
    fig.suptitle(title, fontname=font)

    prepare_heatmap(axd['left'], pu_y_at_t, 'Pickup demand')
    prepare_heatmap(axd['center'], do_y_at_t, 'Dropoff demand')
    prepare_heatmap(axd['right'], delta_at_t, 'Delta')
    prepare_line_plots(axd['bottom'], pu, do, delta, point_in_time, csfont)

    fig.savefig(f'./visualization/graphics/DemandSupply_{str(point_in_time).rjust(3, "0")}.PNG', bbox_inches='tight')
    plt.close()


def prepare_line_plots(ax, pu, do, delta, point_in_time, csfont):

    sns.lineplot(x=list(range(7 * 48)), y=pu, ax=ax, color='royalblue', legend=False)
    sns.lineplot(x=list(range(7 * 48)), y=do, ax=ax, color='mediumseagreen', legend=False)
    sns.lineplot(x=list(range(7 * 48)), y=delta, ax=ax, color='mediumvioletred', legend=False)

    # Axis labels
    ax.set_ylabel("#Trips", **csfont)
    ax.set_xlabel('')

    # Add separation of days
    for i in range(1, 7):
        ax.axvline(i * 48, color='black', alpha=.25)
    ax.axvline(point_in_time, color='black')

    ax.set_xlim(0, 335)

    # Set font for labels
    ax.set_xticks([x * 48 + 24 for x in range(0, 7)])
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    for tick in ax.get_xticklabels(): tick.set_fontname(font)
    for tick in ax.get_yticklabels(): tick.set_fontname(font)

    legend = ax.legend(['Pickup or demand', 'Dropoff or supply', 'Delta'], frameon=True, prop={'family': font},
                       loc='lower left')
    legend.get_frame().set_linewidth(0.0)


def prepare_heatmap(ax, matrix, title='', cmap='coolwarm'):
    sns.heatmap(matrix, cmap=cmap, vmin=-15, vmax=40, center=0, ax=ax)
    ax.set_title(title, fontname=font)
    ax.set_xlabel('x-index of grid', fontname=font)
    ax.set_ylabel('y-index of grid', fontname=font)
    for tick in ax.get_xticklabels(): tick.set_fontname(font)
    for tick in ax.get_yticklabels(): tick.set_fontname(font)
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(0, matrix.shape[0])


def visualize_delta_and_shap_values_and_q_values(readable_time, ppu_at_t, do_at_t, shap_ppu_at_t, shap_do_at_t, taxi_location, action, cmap, q, title):

    taxi_location = taxi_location.tolist()[0]

    fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    fig.suptitle(readable_time, fontname=font, fontsize=14)

    sns.heatmap(ppu_at_t[0, 0].numpy(), center=0, cmap='coolwarm', ax=ax[0], cbar_kws={"shrink": .9})
    ax[0].set_title('Predicted Pickup Demand', fontname=font)
    ax[0].set_xlabel('x-index of grid', fontname=font)
    ax[0].set_ylabel('y-index of grid', fontname=font)
    ax[0].set_xlim(0, 20)
    ax[0].set_ylim(0, 20)
    ax[0].add_patch(
        Rectangle((taxi_location[0] - 2, taxi_location[1] - 2), 5, 5, fill=False, edgecolor='royalblue', lw=1))
    ax[0].add_patch(Rectangle(taxi_location, 1, 1, fill=False, edgecolor='gold', lw=2))
    ax[0].add_patch(
        Rectangle((taxi_location[0] + action[0], taxi_location[1] + action[1]), 1, 1, fill=False, edgecolor='black',
                  lw=3))


    sns.heatmap(do_at_t[0, 0].numpy(), center=0, cmap='coolwarm', ax=ax[1], cbar_kws={"shrink": .9})
    ax[1].set_title('Dropoff Demand', fontname=font)
    ax[1].set_xlabel('x-index of grid', fontname=font)
    ax[1].set_ylabel('y-index of grid', fontname=font)
    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(0, 20)
    ax[1].add_patch(
        Rectangle((taxi_location[0] - 2, taxi_location[1] - 2), 5, 5, fill=False, edgecolor='royalblue', lw=1))
    ax[1].add_patch(Rectangle(taxi_location, 1, 1, fill=False, edgecolor='gold', lw=2))
    ax[1].add_patch(
        Rectangle((taxi_location[0] + action[0], taxi_location[1] + action[1]), 1, 1, fill=False, edgecolor='black',
                  lw=3))

    # SHAP values
    sns.heatmap(shap_ppu_at_t, cmap=cmap, center=0, ax=ax[2], cbar_kws={"shrink": .9})
    ax[2].set_title('SHAP values Predicted PU', fontname=font)
    ax[2].set_xlabel('x-index of grid', fontname=font)
    ax[2].set_ylabel('y-index of grid', fontname=font)
    ax[2].set_xlim(0, 20)
    ax[2].set_ylim(0, 20)
    ax[2].add_patch(
        Rectangle((taxi_location[0] - 2, taxi_location[1] - 2), 5, 5, fill=False, edgecolor='royalblue', lw=1))
    ax[2].add_patch(Rectangle(taxi_location, 1, 1, fill=False, edgecolor='gold', lw=2))
    ax[1].add_patch(
        Rectangle((taxi_location[0] + action[0], taxi_location[1] + action[1]), 1, 1, fill=False, edgecolor='black',
                  lw=3))


    # SHAP values
    sns.heatmap(shap_do_at_t, cmap=cmap, center=0, ax=ax[3], cbar_kws={"shrink": .9})
    ax[3].set_title('SHAP values DO', fontname=font)
    ax[3].set_xlabel('x-index of grid', fontname=font)
    ax[3].set_ylabel('y-index of grid', fontname=font)
    ax[3].set_xlim(0, 20)
    ax[3].set_ylim(0, 20)
    ax[3].add_patch(
        Rectangle((taxi_location[0] - 2, taxi_location[1] - 2), 5, 5, fill=False, edgecolor='royalblue', lw=1))
    ax[3].add_patch(Rectangle(taxi_location, 1, 1, fill=False, edgecolor='gold', lw=2))
    ax[3].add_patch(
        Rectangle((taxi_location[0] + action[0], taxi_location[1] + action[1]), 1, 1, fill=False, edgecolor='black',
                  lw=3))


    # Q-values
    cmap = sns.color_palette("flare", as_cmap=True)
    hm = sns.heatmap(q, cmap=cmap, ax=ax[4], vmin=q.min(), vmax=q.max(), cbar_kws={"shrink": .9})
    hm.add_patch(Rectangle((2 + action[0], 2 + action[1] * -1), 1, 1, fill=False, edgecolor='black', lw=2))
    hm.add_patch(Rectangle((2, 2), 1, 1, fill=False, edgecolor='gold', lw=1))
    ax[4].set_title('Q-values', fontname=font)

    for x in range(4):
        for tick in ax[x].get_xticklabels(): tick.set_fontname(font)
        for tick in ax[x].get_yticklabels(): tick.set_fontname(font)

    fig.savefig(f'./graphics/{title}.PNG', bbox_inches='tight')
    # plt.show()
    plt.close()


def visualize_q_value_arrow_map(matrix, arrows, title, current_t):

    min_q_value = np.asarray([a[2] for a in arrows]).min()
    max_q_value = np.asarray([a[3] for a in arrows]).max()
    min_diff = np.asarray([a[4] for a in arrows]).min()
    max_diff = np.asarray([a[4] for a in arrows]).max()

    sns.heatmap(matrix, cmap='coolwarm', vmin=matrix.min(), vmax=matrix.max(), center=0)
    for a in arrows:
        alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
        plt.arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], alpha=alpha, head_width=0.1)  # alpha=min(a[2]*5, 1)
    plt.title(title, fontname=font)
    plt.xlabel('x-index of grid', fontname=font)
    plt.ylabel('y-index of grid', fontname=font)
    plt.yticks(fontname=font), plt.xticks(fontname=font)
    plt.xlim(0, matrix.shape[1])
    plt.ylim(0, matrix.shape[0])

    plt.text(.5, 20 - 1, f'Min. Q-value: {min_q_value:.2f}', fontname=font, color='white', backgroundcolor='black')
    plt.text(.5, 20 - 2.5, f'Max. Q-value: {max_q_value:.2f}', fontname=font, color='white', backgroundcolor='black')

    plt.savefig(f'./graphics/action-map/action_map_{str(current_t)}.PNG', bbox_inches='tight')
    plt.close()


def visualize_q_value_arrow_map_v2(pu, do, delta, arrows, title, current_t):

    min_diff = np.asarray([a[4] for a in arrows]).min()
    max_diff = np.asarray([a[4] for a in arrows]).max()

    pu_cmap = 'Greens'
    do_cmap = 'PuRd'
    delta_cmap = sns.color_palette('bwr', as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))  # (#plots on height, #plots on width), (width, height)
    fig.suptitle(title, fontname=font, fontsize=14)

    hm = sns.heatmap(pu, cmap=pu_cmap, vmin=0, vmax=65, ax=ax[0], cbar=True, square=True, cbar_kws={"shrink": .9})
    for a in arrows:
        alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
        ax[0].arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], alpha=alpha, head_width=0.1)
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax[0].set_title('Predicted Pick-Up Demand', fontname=font)

    hm = sns.heatmap(do, cmap=do_cmap, vmin=0, vmax=65, ax=ax[1], cbar=True, square=True, cbar_kws={"shrink": .9})
    for a in arrows:
        alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
        ax[1].arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], alpha=alpha, head_width=0.1)
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax[1].set_title('Drop-Off Demand', fontname=font)

    hm = sns.heatmap(delta, cmap=delta_cmap, vmin=-125, center=0, vmax=65, ax=ax[2], cbar=True, square=True, cbar_kws={"shrink": .9})
    for a in arrows:
        alpha = min(0.1 + (a[4] - min_diff) / (max_diff - min_diff), 1)
        ax[2].arrow(a[0][0] + 0.5, a[0][1] + 0.5, a[1][0], a[1][1], alpha=alpha, head_width=0.1)
    hm.set_ylim(0, 20)
    for tick in hm.collections[0].colorbar.ax.get_yticklabels(): tick.set_fontname(font)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    ax[2].set_title('Delta', fontname=font)

    for x in range(3):
        for tick in ax[x].get_xticklabels(): tick.set_fontname(font)
        for tick in ax[x].get_yticklabels(): tick.set_fontname(font)

    plt.savefig(f'./graphics/action-map/action_map_{str(current_t)}.PNG', bbox_inches='tight')
    plt.close()


def create_gif(directory, file_name):
    images = []
    list_of_pngs = os.listdir(directory)
    for file in list_of_pngs:
        images.append(imageio.imread(directory + file))
    imageio.mimsave(directory + file_name, images, duration=0.75)
