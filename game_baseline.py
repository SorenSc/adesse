import json
import logging
import random
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st

import game.commonsv2 as game_commons

st.set_page_config(page_title='Taxi Repositioning Game', page_icon=':v:', layout='wide')
dp_exp_btn_labels = ['A/Taxi', 'B/Recommended', 'C', 'D', 'E', 'F']
table_columns = ['Feature', 'Value', 'Importance']

container_header = st.container()
container_header.title('Taxi Repositioning Game')


def seperate_into_multiple_lines(text):
    letter_regex = "[^\W\d_]"
    result, lines = '', re.findall('.{1,43}', text)
    for i in range(len(lines) - 1):
        ends_with_letter = len(re.findall(letter_regex, lines[i][-1]))
        starts_with_letter = len(re.findall(letter_regex, lines[i+1][0]))
        binding = '-' if (ends_with_letter == 1) & (starts_with_letter == 1) else ' '
        line =  lines[i] + binding + '\n'
        line = line[1:] if line.startswith(' ') else line
        result  += line
    result += lines[-1]
    return result

# Sidebar stores description of various explanations
with st.sidebar:
    st.header('Explanation of the advice')
    text = 'For each cell, the explanation shows the influence of the corresponding value in the state. With influence we refer to the influence the value in the corresponding cell has on the advice.'
    text = seperate_into_multiple_lines(text)
    st.text(text)

    st.header('Previous, current taxi location, and advice')
    text = 'The previous location is marked with a black rectangle, the current one with a yellow one, and the advice with a blue one.'
    text = seperate_into_multiple_lines(text)
    st.text(text)

st.markdown('---')

c_location_and_reward = st.columns(8)
c_location_and_reward[0].text(f'A/Taxi location:')
c_location_content = c_location_and_reward[1].markdown(f'')
c_location_and_reward[2].text(f'Last reward:')
c_last_reward_content = c_location_and_reward[3].markdown(f'')
c_location_and_reward[4].text(f'Acc. reward:')
c_acc_reward_content = c_location_and_reward[5].markdown(f'')
c_location_and_reward[6].text(f'Remaining steps:')
c_remaining_steps = c_location_and_reward[7].markdown(f'')

st.markdown('---')

container_explanation = st.container()
container_explanation.text('Explanation of the advice') 
c_explanation_columns = container_explanation.columns([2.65, 2, 2, 2, 2])

st.markdown('---')

container_state = st.container()
container_state.text('State') 
c_state_columns = container_state.columns([2.65, 2, 2, 2, 2])

if 'run_id' not in st.session_state:
    st.session_state.run_id = random.randint(10000, 99999)
if 'proposed_action' not in st.session_state:
    st.session_state.proposed_action = [0, 0]
if 'taxi_location' not in st.session_state:
    st.session_state.taxi_location = [0, 0]
if 'last_location' not in st.session_state:
    st.session_state.last_location = [0, 0]
if 'last_reward' not in st.session_state:
    st.session_state.last_reward = ''
if 'accumulated_reward' not in st.session_state:
    st.session_state.accumulated_reward = 0
if 'remaining_steps' not in st.session_state:
    st.session_state.remaining_steps = 13
if 'impossible_actions' not in st.session_state:
    st.session_state.impossible_actions = ''
if 'dp_exp_inaccessible_btns' not in st.session_state:
    # Make all inaccessible in the first place as there is no explanation available.
    st.session_state.dp_exp_inaccessible_btns = '; '.join(dp_exp_btn_labels)
if 'dp_exp_seleted_btn' not in st.session_state:
    st.session_state.dp_exp_seleted_btn = 1
if 'explanation' not in st.session_state:
    st.session_state.explanation = {}
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

vis_elements = {
    'state0': c_state_columns[0],
    'state1': c_state_columns[1],
    'state2': c_state_columns[2],
    'state3': c_state_columns[3],
    'state4': c_state_columns[4],
    'explanation0': c_explanation_columns[0],
    'explanation1': c_explanation_columns[1],
    'explanation2': c_explanation_columns[2],
    'explanation3': c_explanation_columns[3],
    'explanation4': c_explanation_columns[4],
    'location': c_location_content,
    'last_reward': c_last_reward_content,
    'acc_reward': c_acc_reward_content,
    'remaining_steps': c_remaining_steps,
}


def visualize(type, vis_elements, action_x, action_y, dp_exp_selected_btn=None):
    logging.info(f'Visualize with type: {type}')
    session = requests.Session()
    result = json.loads(session.get('http://127.0.0.1:5001/step/',
        params={
            'run_id': st.session_state.run_id, 'action_x': action_x, 'action_y': action_y
        }).json())
    st.session_state.proposed_action = result['proposed_action']
    st.session_state.taxi_location = result['taxi_location']
    st.session_state.last_location = result['last_location']
    st.session_state.last_reward = result['last_reward']
    st.session_state.accumulated_reward = result['accumulated_reward']
    if result['last_reward'] != 20:
        st.session_state.remaining_steps -= 1
    else:
        st.session_state.remaining_steps -= 2
    st.session_state.impossible_actions = result['impossible_actions']
    st.session_state.explanation = result['explanation']

    # Create state plots
    st.session_state.state_figure4 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['r_X'][0]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 0, True, 'state', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_r_X_0.PNG')
    st.session_state.state_figure3 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['r_X'][1]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 1, True, 'state', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_r_X_1.PNG')
    st.session_state.state_figure2 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['r_X'][2]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 2, True, 'state', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_r_X_2.PNG')
    st.session_state.state_figure1 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['r_X'][3]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 3, True, 'state', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_r_X_3.PNG')
    st.session_state.state_figure0 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['t_y']), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 4, True, 'state', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_t_y.PNG')

    # Create explanation plots
    st.session_state.exp_figure4 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['lime'][0]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 0, True, 'lime', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_lime_0.PNG')
    st.session_state.exp_figure3 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['lime'][1]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 1, True, 'lime', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_lime_1.PNG')
    st.session_state.exp_figure2 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['lime'][2]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 2, True, 'lime', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_lime_2.PNG')
    st.session_state.exp_figure1 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['lime'][3]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 3, True, 'lime', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_lime_3.PNG')
    st.session_state.exp_figure0 = game_commons.visualize_baseline_explanation(
            np.asarray(st.session_state.explanation['lime'][4]), st.session_state.last_location, st.session_state.taxi_location,
            st.session_state.proposed_action, 4, True, 'lime', f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_BL_lime_4.PNG')

    # Visualize the status
    vis_elements['location'].markdown(f'**[{st.session_state.taxi_location[0]}, {st.session_state.taxi_location[1]}]**')
    vis_elements['last_reward'].markdown(f'**{st.session_state.last_reward}**')
    vis_elements['acc_reward'].markdown(f'**{st.session_state.accumulated_reward}**')
    vis_elements['remaining_steps'].markdown(f'**{st.session_state.remaining_steps}**')

    # Visualize state plots
    if vis_elements['state0'] is not None: vis_elements['state0'].empty()
    vis_elements['state0'] = vis_elements['state0'].pyplot(st.session_state.state_figure0)
    if vis_elements['state1'] is not None: vis_elements['state1'].empty()
    vis_elements['state1'] = vis_elements['state1'].pyplot(st.session_state.state_figure1)
    if vis_elements['state2'] is not None: vis_elements['state2'].empty()
    vis_elements['state2'] = vis_elements['state2'].pyplot(st.session_state.state_figure2)
    if vis_elements['state3'] is not None: vis_elements['state3'].empty()
    vis_elements['state3'] = vis_elements['state3'].pyplot(st.session_state.state_figure3)
    if vis_elements['state4'] is not None: vis_elements['state4'].empty()
    vis_elements['state4'] = vis_elements['state4'].pyplot(st.session_state.state_figure4)

    # Visualize explanation plots
    if vis_elements['explanation0'] is not None: vis_elements['explanation0'].empty()
    vis_elements['explanation0'] = vis_elements['explanation0'].pyplot(st.session_state.exp_figure0)
    if vis_elements['explanation1'] is not None: vis_elements['explanation1'].empty()
    vis_elements['explanation1'] = vis_elements['explanation1'].pyplot(st.session_state.exp_figure1)
    if vis_elements['explanation2'] is not None: vis_elements['explanation2'].empty()
    vis_elements['explanation2'] = vis_elements['explanation2'].pyplot(st.session_state.exp_figure2)
    if vis_elements['explanation3'] is not None: vis_elements['explanation3'].empty()
    vis_elements['explanation3'] = vis_elements['explanation3'].pyplot(st.session_state.exp_figure3)
    if vis_elements['explanation4'] is not None: vis_elements['explanation4'].empty()
    vis_elements['explanation4'] = vis_elements['explanation4'].pyplot(st.session_state.exp_figure4)

st.markdown('---')

container_actions = st.container()
container_actions.markdown(f"**Please select one of the actions by clicking on the corresponding button!**")

# Create action buttons
for y_index in range(5):
    action_container = container_actions.columns(12)
    for x_index in range(5):
        label = f'{x_index - 2}, {(y_index - 2) * -1}'
        if action_container[x_index + 0].button(label, key=label):  # Create action buttons
            visualize('step', vis_elements, x_index - 2, (y_index - 2) * -1)

container_actions.text(f"The blue button is the advice; the yellow your current location.")
container_actions.text("The button '-1,0' refers to moving one cell to the left or -1 steps on the x-axis\n"
                       "and 0 steps on the y-axis.")

# Avoides a reset button
if st.session_state.first_run:
    st.session_state.first_run = False
    visualize('step', vis_elements, 99, 99)  # Reset the environment

st.markdown('---')

game_commons.mark_action_buttons(st.session_state.proposed_action, st.session_state.taxi_location)
game_commons.disable_impossible_actions(st.session_state.impossible_actions)