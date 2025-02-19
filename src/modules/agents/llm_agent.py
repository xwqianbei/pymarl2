import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm



class LLMAgent(nn.Module):
    """agent trajectory encoder and policy network

    Attributes:
        input_shape: the agent observation shape
        args: get from the algs config.yaml
    
    Returns:
        q: the logits of the each action
        agent_hh: the hidden state of the GRUCell
    """
    def __init__(self, input_shape: th.tensor, args: dict):
        """Build the agent trajectory encoder and embedding decoder

        Args:
            agent trajectory encoder: fc1 + rnn
            embedding decoder: two fc layers
        """
        super(LLMAgent, self).__init__()
        self.args = args

        # agent trajectory encoder
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # role decoder to generate the parameters of the policy network
        self.role_emb_dec_weight = nn.Sequential(
            nn.Linear(args.role_embedding_dim, args.role_emb_dec_hid_dim),
            nn.ReLU(),
            nn.Linear(args.role_emb_dec_hid_dim, args.rnn_hidden_dim * args.n_actions)
        )
        self.role_emb_dec_bias = nn.Sequential(
            nn.Linear(args.role_embedding_dim, args.role_emb_dec_hid_dim),
            nn.ReLU(),
            nn.Linear(args.role_emb_dec_hid_dim, args.n_actions)
        )


    def forward(self, inputs, agent_hid_in, role_embedding):
        """forward pass of the agent

        Args:
            inputs: the agent observation
            agent_hidden_state: the hidden state of the GRUCell
            role_embedding: the role embedding of the agent
        
        Returns:
            q: the logits of the each action
            agent_hh: the hidden state of the GRUCell
        """

        bs, a, e = inputs.size()
        inputs = inputs.reshape(-1, e)
        agent_hid_in = agent_hid_in.reshape(-1, self.args.rnn_hidden_dim)
        role_embedding = role_embedding.reshape(-1, self.args.role_embedding_dim)
        
        # [bs * a, rnn_hidden_dim]
        x = F.relu(self.fc1(inputs), inplace = True)
        agent_hid_out = self.rnn(x, agent_hid_in)

        # [bs * a, rnn_hidden_dim, n_actions]
        role_emb_dec_weight = self.role_emb_dec_weight(role_embedding).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        bias = self.role_emb_dec_bias(role_embedding).reshape(-1, self.args.n_actions)  

        # [bs * a, n_actions]
        q = th.bmm(agent_hid_out.unsqueeze(1), role_emb_dec_weight).squeeze(1) + bias

        return q.reshape(bs, a, -1), agent_hid_out.reshape(bs, a, -1)


        
    