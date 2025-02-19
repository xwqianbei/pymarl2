import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Traj_hid_alignment(nn.Module):
    """Align the traj_hid_vector to the thought text embedding
        Attributes:
            hidden_dim: the hidden dimension of the trajectory encoder
            thought_text_dim: the dimension of the thought text embedding
        
        Returns:
            thought_text_embedding: the thought text embedding
    """

    def __init__(self, args):
        self.decoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.role_embedding_dim),
            nn.Tanh()
        )

    def forward(self, traj_hid_state):
        b, a, e = traj_hid_state.size()

        traj_hid_state = traj_hid_state.reshape(-1, e)
        traj_align_out = self.decoder(traj_hid_state)
        return traj_align_out.reshape(b, a, -1)