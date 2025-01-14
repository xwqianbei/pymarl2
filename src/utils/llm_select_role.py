from .prompt_template import *
from .chat_with_gpt import callgpt


class SC2_select_role(object):
    # TODO: add num_agent
    def __init__(self, env_name, map_name, num_agent, role_num):
        self.env_name = env_name
        self.map_name = map_name
        self.num_agent = num_agent
        self.role_num = role_num

    def parse_gpt_output(self, llm_output):
        # llm_output:[dict] [num_agent]
        ROLE2ONEHOT = {
            "Focus Fire": [1, 0, 0, 0],
            "Retreat": [0, 1, 0, 0],
            "Spread Out": [0, 0, 1, 0],
            "Advance": [0, 0, 0, 1]
        }
        agents_role_one_hot = []
        for agent_res in llm_output:
            agents_role_one_hot.append(ROLE2ONEHOT[agent_res['skill']])
        # [num_agent, role_num]
        return agents_role_one_hot


    def __call__(self, batch_states):
        # TODO: 写一个多线程？
        batch_role_one_hot = []
        for env_states in batch_states:
            llm_output = callgpt(self.env_name, self.map_name, env_states, self.num_agent)
        
            # [num_agent, role_num]
            agents_role_one_hot = self.parse_gpt_output(llm_output)
            batch_role_one_hot.append(agents_role_one_hot)

        # [bs, num_agent, role_num]
        return batch_role_one_hot
    
    