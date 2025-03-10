import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class LLMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LLMAgent, self).__init__()
        self.args = args
        self.role_embeddings = []
        self.role_embedding = None

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.role_selector = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.role_selector_dim),
            nn.ReLU(),
            nn.Linear(args.role_selector_dim, args.role_num)
        )

        self.role_decoder_w = nn.sequential(
            nn.Linear(args.role_embedding_dim, args.role_decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.role_decoder_hidden_dim, args.rnn_hidden_dim * args.n_actions)
        )

        self.role_decoder_b = nn.sequential(
            nn.Linear(args.role_embedding_dim, args.role_decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.role_decoder_hidden_dim, args.n_actions)
        )

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, env_t):
        """
        Args:
            inputs: [bs, a, e]
            hidden_state: [bs, rnn_hidden_dim]
            env_t: int
        Returns:
            q_val: [bs, a, n_actions]
            hh_out: [bs, a, rnn_hidden_dim]
            role_probs: [bs * a, role_num]        
        """
        bs, a, e = inputs.size()

        inputs = inputs.reshape(-1, e) # [bs * a, e]

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in) # [bs * a, rnn_hidden_dim]
        

        role_logits = self.role_selector(hh) # [bs * a, role_num]
        role_probs = F.softmax(role_logits, dim=-1) # [bs * a, role_num]
        role_idx = th.argmax(role_probs, dim=-1)
        if env_t % self.args.role_change_interval == 0:
            self.role_embedding = self.role_embeddings[role_idx]

        policy_w = self.role_decoder_w(self.role_embedding).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions) # [bs * a, rnn_hidden_dim * n_actions]
        policy_b = self.role_decoder_b(self.role_embedding) # [bs * a, n_actions]

        q_val = th.bmm(hh.unsqueeze(1), policy_w).squeeze(1) + policy_b # [bs * a, n_actions]

        return q_val.reshape(bs, a, -1), hh.reshape(bs, a, -1), role_probs.reshape(bs, a, -1)
     

         

        



        