from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from utils.llm_select_role import SC2_select_role


class LLMMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(LLMMAC, self).__init__(scheme, groups, args):
    
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
        role_one_hot = SC2_select_role(curr_states, self.args.role_num)

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, role_one_hot)
        
        return agent_outs