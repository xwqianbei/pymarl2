from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from modules.layer.role_selector import RoleSelector
from utils.text_embedding import text_embedding

# TODO: fix code
class LLMMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(LLMMAC, self).__init__(scheme, groups, args)
        self.role_embeddings = text_embedding(args.role_desc_set, args.embedding_model_path)
        self.role_hidden_states = None

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
        
        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs, self.hidden_states, self.role_hidden_states = self.agent(agent_inputs, self.hidden_states, self.role_hidden_states, self.role_embeddings)

        return agent_outs