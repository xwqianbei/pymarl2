import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class RoleSelector(nn.Module):
    """
    To assign a role to each agent in the team
    And return the role embedding of the role
    """

    def __init__(self, input_shape, role_embeddings, args):
        super(RoleSelector, self).__init__()
        self.args = args
        self.role_embeddings = role_embeddings

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.role_num)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e) # [b * a, e]
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)
        
        role_probs = F.softmax(self.fc2(hh), dim = 1)
        role_indices = th.argmax(role_probs, dim = -1)
        role_onehot = F.one_hot(role_indices, self.args.role_num)

        # TODO: Check if this is correct
        role_embedding = self.role_embeddings[role_indices]
        return role_onehot, role_embedding, hh.reshape(b, a, -1)

class LLMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LLMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.fc2_b = nn.Linear(args.role_embedding_dim, args.rnn_hidden_dim)
        self.fc2_w = nn.Linear(args.role_embedding_dim, args.rnn_hidden_dim * args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, role_embedding):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e) # [b * a, e]
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in) # [bs, rnn_hidden_dim]

        # w: [bs, rnn_hidden_dim,, n_action]
        w = self.fc2_w(role_embedding).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        b = self.fc2_b(role_embedding).reshape(-1, self.args.rnn_hidden_dim)

        # [bs, n_action]
        q = th.bmm(hh.unsqueeze(1), w).squeeze(1) + b
        return q.view(b, a, -1), hh.view(b, a, -1)

        