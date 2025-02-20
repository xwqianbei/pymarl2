from modules.layer.role_selector import RoleSelector
import torch as th
import numpy as np



class RoleController(object):
    """RoleSelector wraper
        Attributes:
            role_selector: the role selector to assign roles to agents
            hidden_states: the hidden state of the role selector
        
        select_role: select max probability role
        forward: forward the inputs to the role selector
    """

    def __init__(self, scheme, args):
        self.args = args

        input_shape = scheme["state"]["vshape"]
        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.obs_agent_id:
            input_shape += self.n_agents

        self.role_selector = RoleSelector(input_shape, args)
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.role_selector.init_hidden(batch_size)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.args.n_agents, -1)  # bav

    def select_role(self, inputs):
        role_probs = self.forward(inputs, test_mode=True)
        role_indices = th.argmax(role_probs, dim = -1)
        return role_indices

    def forward(self, inputs, test_mode=False):
        if test_mode:
            self.role_selector.eval()
        
        role_probs, self.hidden_states = self.role_selector(inputs, self.hidden_states)
        return role_probs
        