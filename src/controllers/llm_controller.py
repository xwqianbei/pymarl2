from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from utils.llm_select_role import SC2_select_role
from envs import REGISTRY as env_REGISTRY


class LLMMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(LLMMAC, self).__init__(scheme, groups, args)
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]


        # TODO: check the curr_states and role_one_hot dimensions
        curr_states = self._build_states(ep_batch, t)
        # TODO: 对齐调整后的参数
        
        # [env_name, map_name, num_agent, role_num]
        env_fn = env_REGISTRY[self.args.env]
        env = env_fn(**self.args.env_args)
        role_selector = SC2_select_role(self.args.env, self.args.env_args.map_name, env.n_agents, self.args.role_num)
        
        role_one_hot = role_selector(curr_states)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, role_one_hot)
        
        return agent_outs