
class SC2_prompt(object):
    def __init__(self, map_name="5m_vs_6m")->None:
        self.map_name = map_name
        self.task_description = f"We are playing StarCraft II micro scenario, tring to control our agents to defeat all of the enemy units.\n"
        self.state_form = "In each step, the current state is a 1-dimensional list: [nf_al]*n_agents + [nf_en]*n_enemies + [last_actions].\
nf_al denotes the unit state for each agent with attributes [health_rate, weapon_cooldown_rate or energy_rate, relative_x_to_map_center, \
relative_y_to_map_center, shield_rate(1 dimension if a_race is P else 0 dimension), unit_type_bits(the dimension is defined in the map config)]. \
nf_en represents the unit state for each enemy with attributes [health_rate, relative_x_to_map_center, \
relative_y_to_map_center, shield_rate(1 dimension if b_race in map config is P else 0 dimension), unit_type_bits(the dimension is defined in the map config)]. \
The last_actions doesn't require consideration.\n"
        self.role_instruction = f"Your role is to dynamically assign one of the following skills to the agent based on the current state:\n\
1. Focus Fire 2. Retreat 3. Spread Out 4. Advance 5. Dead\n\
Please adhere to the following guidelines:\n\
1. Use only the given state information and  without relying on any unspecified details!\n\
2. Assign skills that align with the battlefield context and the current state of each agent or enemy.\n\
3. Only assign 'Dead' when the agent health is 0.\n\
4. Provide clear conditions for assigning each skill!\n\
5. Provide the thought process for assigning skills to each agent.\n\
6. Provide exclusively a JSON-formatted string. Avoid any extra text or outputs.\n\
Please respond in the following JSON format:\n" + \
"""
[
    {
        'agentID': '0',
        'skill': 'Focus Fire',
        'conditions': 'the conditions for assigning each skill.',
        'thought_process': 'the thought process for assigning skills to each agent.',
    },
    {
        'agentID': '1',
        'skill': 'Advance',
        'conditions': 'the conditions for assigning each skill.',
        'thought_process': 'the thought process for assigning skills to each agent.',
    },
    ...
]
"""
    def get_message(self, env_states):
        message=[]
        message.append({'role':'system', 'content':self.task_description + self.state_form + self.role_instruction})
        # TODO: process the self.map_config
        message.append({'role':'user', 'content':f"Task is {self.map_name}.\ncurrent_state is {env_states}."})
        return message
    