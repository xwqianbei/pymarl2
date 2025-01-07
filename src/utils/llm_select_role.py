from .prompt_template import *
from .chat_with_gpt import *

class SC2_select_role(object):
    def __init__(self):
        pass


    def gpt_json_load(json_str):
        json_str = json_str.replace("```json", "").replace("```", "").replace("\n", "").strip()
        json_data = json.loads(json_str, strict=False)
        if isinstance(json_data, str):
            return json.loads(json_data)
        return json_data

    def __call__(batch_states, role_num):
        pass
        
        # return [bs, n_agents, role_num]
