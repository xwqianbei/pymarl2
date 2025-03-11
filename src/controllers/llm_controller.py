from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

class LLMMAC(BasicMAC):
    """The multi-agent controller for the LLM architecture. This controller shares parameters between agents.
    Args:
        scheme: The scheme for the input and output tensors.
        groups: The groups for the agents.
        args: The arguments for the controller.
    """
    def __init__(self, scheme, groups, args):
        super(LLMMAC, self).__init__(scheme, groups, args)
        
    # 
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions for the agents in the batch.

        """
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode = False):
        """Forward pass for the controller.
        - Returns:
            agent_outs: The q_vals of agents
            hidden_states: The trajectory's hidden states
            role_probs: The role probs
            traj_transfer_embd: The trajectory transfer embedding
        """
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, role_probs, traj_transfer_embd = self.agent(agent_inputs, self.hidden_states, t)

        return agent_outs, role_probs, traj_transfer_embd

    