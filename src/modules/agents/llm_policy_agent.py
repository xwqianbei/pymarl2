import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class LLMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LLMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.role_encoder = nn.Linear(args.role_num, args.role_embedding_dim)
        self.role_decoder_w = nn.Linear(args.role_embedding_dim, args.rnn_hidden_dim * args.n_actions)
        self.role_decoder_b = nn.Linear(args.role_embedding_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.role_encoder)
            orthogonal_init_(self.role_decoder_w)
            orthogonal_init_(self.role_decoder_b)
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state, role_one_hot):
        # [bs, n_agents, obs_dim]
        b, a, e = inputs.size()
        # TODO: Note the role_ont_hot dimension
        role_one_hot = role_one_hot.view(-1, self.args.role_num)

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        # TODO: need Relu?
        role_embedding = F.relu(self.role_encoder(role_one_hot), inplace=True)
        w = self.role_decoder_w(role_embedding)
        b = self.role_decoder_b(role_embedding)

        if getattr(self.args, "use_layer_norm", False):
            q = th.bmm(self.layer_normal(hh).unsqueeze(1), w.view(b, a, -1).view(b, -1, a)).squeeze(1) + b
        else:
            q = th.bmm(hh.unsqueeze(1), w.view(b, a, -1).view(b, -1, a)).squeeze(1) + b

        return q.view(b, a, -1), hh.view(b, a, -1)
    

        
