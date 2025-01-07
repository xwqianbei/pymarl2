from envs import REGISTRY as env_REGISTRY

class SC2_prompt(object):
    def __init__(self, map_name="5m_vs_6m")->None:
        self.map_name = map_name
        self.env = env_REGISTRY['sc2'](map_name=self.map_name)
        self.env.reset()
        from envs.starcraft.smac_maps import get_map_params
        self.map_config = get_map_params(map_name)

        self.task_description = f"We are playing StarCraft II micro scenario, tring to control our agents to defeat all of the enemy units.\n"
        self.state_form = "In each step, the current state is a 1-dimensional list: [nf_al]*n_agents + [nf_en]*n_enemies + [last_actions].\
nf_al denotes the unit state for each agent with attributes [health_rate, weapon_cooldown_rate or energy_rate, relative_x_to_map_center, \
relative_y_to_map_center, shield_rate(1 dimension if a_race is P else 0 dimension), unit_type_bits(the dimension is defined in the map config)]. \
nf_en represents the unit state for each enemy with attributes [health_rate, relative_x_to_map_center, \
relative_y_to_map_center, shield_rate(1 dimension if b_race in map config is P else 0 dimension), unit_type_bits(the dimension is defined in the map config)]. \
The last_actions doesn't require consideration.\n"
        self.role_instruction = f"Your role is to assign dynamic roles (e.g., Attacker, Supporter, Scout, etc.) to each agent based on the current state. Please adhere to the following guidelines:\n\
1. Use only the given state information and  without relying on any unspecified details!\n\
2. Assign roles that align with the battlefield context and the current state of each agent or enemy.\n\
3. The roles may include predefined roles such as 'attacker', 'supporter', and 'scout' or can involve other context-driven roles that you define.\n \
4. Provide clear conditions for assigning each role!\n\
5. Provide the thought process for assigning roles to each agent.\n\
6. Provide exclusively a JSON-formatted string compatible with Python's json.loads for parsing. Avoid any extra text or outputs.\n\
Please respond in the following JSON format:\n" + \
"""
{
    'agentID':{
        'role': 'role_name',
        'conditions': 'the conditions for assigning each role.',
        'thought_process': 'the thought process for assigning roles to each agent.',
    },
}
"""
    def get_message(self, env_states):
        message=[]
        message.append({'role':'system', 'content':self.task_description + self.state_form + self.role_instruction})
        message.append({'role':'user', 'content':f"Task is {self.map_name}. The map config is {str(self.map_config)}.\ncurrent_state is {env_states}."})
        return message

