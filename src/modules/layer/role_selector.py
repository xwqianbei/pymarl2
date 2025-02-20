import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RoleSelector(nn.Module):
    """Assign a role to each agent
        Attributes:
            input_shape: the observations' shape
            args: the arguments from the algs config.yaml
        Returns:
            role_probs: the role probabilities of the agents
    """

    def __init__(self, input_shape, args):
        super(RoleSelector, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.role_sel_hid_dim)
        self.rnn = nn.GRUCell(args.role_sel_hid_dim, args.role_sel_hid_dim)
        self.fc2 = nn.Linear(args.role_sel_hid_dim, args.role_num)

    def init_hidden(self, batch_size):
        return self.fc1.weight.new(1, self.args.role_sel_hid_dim).zero_()

    def forward(self, inputs, role_hid_in):
        bs, a, e = inputs.size()
        inputs = inputs.reshape(-1, e)
        role_hid_in = role_hid_in.reshape(-1, self.args.role_sel_hid_dim)

        x = F.relu(self.fc1(inputs))
        role_hid_out = self.rnn(x, role_hid_in)
        logits = self.fc2(role_hid_out)
        role_probs = F.softmax(logits, dim = -1)

        return role_probs.reshape(bs, a, -1), role_hid_out.reshape(bs, a, -1)
    



class RoleSelectorBaseHid(nn.Module):
    """Assign a role to each agent
        Attributes:
            input_shape: the trajectory encoder' hidden_state or the observations' shape
            args: the arguments from the algs config.yaml
        Returns:
            role_probs: the role probabilities of the agents
    """

    def __init__(self, input_shape, args):
        super(RoleSelectorBaseHid, self).__init__()
        self.args = args

        self.role_selector = nn.Sequential(
            nn.Linear(input_shape, args.role_sel_hid_dim),
            nn.ReLU(),
            nn.Linear(args.role_sel_hid_dim, args.role_num)
        )
    
    # TODO: check the inputs shape [bs * a, e]
    def forward(self, inputs):
        bs, a, e = inputs.size()
        inputs = inputs.reshape(-1, e)

        logits = self.role_selector(inputs)
        role_probs = F.softmax(logits, dim = -1)
        return role_probs.reshape(bs, a, -1)



        