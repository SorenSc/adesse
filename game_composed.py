import json
import logging
import random
import re
import pandas as pd
import requests
import streamlit as st

import game.commonsv2 as game_commons

st.set_page_config(page_title='Taxi Repositioning Game', page_icon=':v:', layout='wide')
dp_exp_btn_labels = ['A/Taxi', 'B/Advice', 'C', 'D', 'E', 'F']
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
    st.header('Taxi Index')
    text = 'The Taxi Index combines the #taxis and the estimated #requests in one grid cell. It is calculated via two things: (1) The ratio between the estimated #requests and the #taxis as well as (2) the ratio between the estimated #requests and the mean #requests to include how much is going on in a cell.'
    text = seperate_into_multiple_lines(text)
    st.text(text)

    st.header('Arrows')
    text = 'The arrows show the most promising advice from the repositioners perspective for each possible location in the grid. The darker the arrow, the more import is taking the action at that location.'
    text = seperate_into_multiple_lines(text)
    st.text(text)

    st.header('Table')
    text = 'As the #request per cell is not known in the next 10 minutes, it is estimated via a model. The features influencing the estimation the most, are shown in the table.'
    text = seperate_into_multiple_lines(text)
    st.text(text)

    st.header('Current taxi location, advice, upcoming locations')
    text = 'The current location is marked with a yellow rectangle, the advice with a blue one, and the ones after that with purple ones.'
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

container_state = st.container()
c_state_header_col1, _, c_state_header_col2 = container_state.columns([4, .5, 3])
c_state_header_col2.text(f'Explanation of request estimation model via\nmost important features for the selected location') 
c_state_col1, _, c_state_col2, c_state_col3 = container_state.columns([4, .5, 1, 2])

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

container_dp_exp_content = c_state_col3.empty()

vis_elements = {
    'explanation': c_state_col1,
    'dp_explanation': container_dp_exp_content,
    'location': c_location_content,
    'last_reward': c_last_reward_content,
    'acc_reward': c_acc_reward_content,
    'remaining_steps': c_remaining_steps,
}


def visualize(type, vis_elements, action_x, action_y, dp_exp_selected_btn=None):
    logging.info(f'Visualize with type: {type}')
    session = requests.Session()
    if type != 'dp_exp_selection':
        if type == 'step':
            result = json.loads(session.get('http://127.0.0.1:5000/step/',
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
        else:
            st.session_state.run_id = random.randint(10000, 99999)
            result = json.loads(session.get('http://127.0.0.1:5000/reset/',
                params={'run_id': st.session_state.run_id}).json())

        # Create plot
        st.session_state.figure = game_commons.visualize_compositional_explanation(
                st.session_state.explanation,
                st.session_state.last_location,
                st.session_state.taxi_location,
                st.session_state.proposed_action,
                False, 
                f'{st.session_state.run_id}_{13 - st.session_state.remaining_steps:02d}_COMPOSED.PNG'
        )

        # Visualize table
        if vis_elements['dp_explanation'] is not None:
            dp_exp_index = 1 if len(st.session_state.explanation['table']) > 1 else 0
            df = pd.DataFrame(
                    data=st.session_state.explanation['table'][dp_exp_index],
                    columns=table_columns)
            # vis_elements['dp_explanation'].dataframe(df.style.pipe(game_commons.table_styler))
            vis_elements['dp_explanation'].dataframe(df[table_columns[0]])
            st.session_state.dp_exp_inaccessible_btns = '; '.join(dp_exp_btn_labels[len(st.session_state.explanation['table']):])
            game_commons.disable_impossible_dp_exp_buttons(st.session_state.dp_exp_inaccessible_btns)

    else:
        session.post('http://127.0.0.1:5000/request_estimation_exp_selection/', params={'selection': dp_exp_selected_btn})
        st.session_state.dp_exp_seleted_btn = dp_exp_selected_btn

    # Visualize the status
    vis_elements['location'].markdown(f'**[{st.session_state.taxi_location[0]}, {st.session_state.taxi_location[1]}]**')
    vis_elements['last_reward'].markdown(f'**{st.session_state.last_reward}**')
    vis_elements['acc_reward'].markdown(f'**{st.session_state.accumulated_reward}**')
    vis_elements['remaining_steps'].markdown(f'**{st.session_state.remaining_steps}**')

    # Visualize plot
    if vis_elements['explanation'] is not None: vis_elements['explanation'].empty()
    vis_elements['explanation'] = vis_elements['explanation'].pyplot(st.session_state.figure)

if vis_elements['dp_explanation'] is not None:
    for btn_index in range(len(dp_exp_btn_labels)):
        if c_state_col2.button(dp_exp_btn_labels[btn_index]):
            df = pd.DataFrame(
                    data=st.session_state.explanation['table'][btn_index],
                    columns=table_columns)
            # vis_elements['dp_explanation'].dataframe(df.style.pipe(game_commons.table_styler))
            vis_elements['dp_explanation'].dataframe(df[table_columns[0]])
            visualize('dp_exp_selection', vis_elements, 0, 0, btn_index)

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
game_commons.mark_dp_explanation_buttons(dp_exp_btn_labels[st.session_state.dp_exp_seleted_btn])
game_commons.disable_impossible_dp_exp_buttons(st.session_state.dp_exp_inaccessible_btns)
