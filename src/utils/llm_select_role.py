from .prompt_template import *
from .chat_with_gpt import callgpt

class SC2_select_role(object):
    def __init__(self):
        pass

    def parse_gpt_output(self, llm_output, role_num):
        pass
    # return [agents, role_num]


    def __call__(self, env_name, map_name, batch_states, role_num):
        # TODO: 写一个多线程？
        batch_role_one_hot = []
        for env_states in batch_states:
            llm_output = callgpt(env_name, map_name, env_states)
        
            # [agents, role_num]
            agents_role_one_hot = self.parse_gpt_output(llm_output, role_num)
            batch_role_one_hot.append(agents_role_one_hot)

        return batch_role_one_hot

        
        # return [bs, n_agents, role_num]
