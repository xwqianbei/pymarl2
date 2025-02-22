import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
# from modules.layer.role_selector import RoleSelector
from modules.layer.traj_hid_alignment import Traj_hid_alignment
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.text_embedding import TextEmbedding
from utils.api import Call_API
from utils.prompt_template import SC2_prompt

from controllers.role_controller import RoleController
class LLMLearner:
    """The learner to train the llm agent

        Attributes:
            mac: agent controller
            scheme: the observation and action space of the agent
            logger: the logger to record the training process
            args: get from the algs config.yaml
        
        Returns:

    """
    def __init__(self, mac, scheme, logger, args):
        """Build the Qmix/qatten/... learner
            Args:
                mac: agent controller
                scheme: the observation and action space of the agent
                logger: the logger to record the training process
                args: get from the algs config.yaml
        """
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        # TODO: add the role_selector/traj_hid_alignment/role_embeddings
        # get the role_selector
        self.role_controller = RoleController(scheme, args)
        self.role_params = list(self.role_controller.parameters())

        # get the traj_hid_alignment
        self.traj_hid_alignment = Traj_hid_alignment(args)
        # TODO: check the optimizer how to do
        self.params += list(self.traj_hid_alignment.parameters())

        # get the candidate role embeddings
        self.prompter = SC2_prompt(map_name=args.map_name)
        self.llmer = Call_API(args)
        self.text_ember = TextEmbedding(self.args.embedding_model_path)
        self.role_embeddings = self.text_ember.embedding_text(self.args.role_desc_set)
        self.role_embedding = None

        # TODO: check the role_optimizer
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.role_optimizer = Adam(params=self.role_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.role_optimizer = RMSprop(params=self.role_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
 
        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def role2onehot(self, role_str: str)->th.Tensor:
        """Convert the role string to one hot tensor
            Args:
                role_str: the role string
            Returns:
                onehot: the one hot tensor of the role string
        """
        onehot = th.zeros(self.args.role_num)
        onehot[self.args.role_desc_set.index(role_str)] = 1
        return onehot

    # TODO: fix the args.num_agetnts
    def llm_labeler(self, batch, t):
        """Assign roles to agents
            Args:
                batch: the batch of the episode
                t: the time step of the episode
            Returns:
                role_labels: the role labels of the agents
                thoughts_emb: the embedding of the thoughts
                thoughts_str: the thoughts of the agents
        """
        states = batch["state"][:, t] # [bs, state_dim]
        states_np = states.detach().cpu().numpy() # [bs, state_dim]

        role_labels = [] # [bs, n_agents, role_num]
        thoughts_str = [] # [bs, n_agents]
        thoughts_emb = [] # [bs, n_agents, emb_dim]
        for d in states_np:
            message = self.prompter.get_prompt(d)
            response_format = {"type": "text"}
            # List[dict], keys: agent ID, skill, conditions, thought_process
            response = self.llmer(message, self.args.num_agents, response_format) # [n_agents]
            
            role_labels.append([r["skill"] for r in response])
            thoughts_str.append([r["thought_process"] for r in response])
            thoughts_emb.append([self.text_ember.embedding_text(r["thought_process"]) for r in response])

        return th.tensor(role_labels), th.tensor(thoughts_emb), thoughts_str

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        # TODO: record info to caculate the loss
        # Assign roles to agents
        self.role_controller.train()
        self.role_controller.init_hidden(batch.batch_size)

        self.traj_hid_alignment.train()

        role_probs = []
        role_llm_labels = []

        # record the trajectory hidden states
        traj_hidden_states = []
        traj_hid_align_outs = []
        traj_thought_strs = []
        traj_thought_embs = []

        # TODO: add the llm labels, llm thoughts, llm embs
        for t in range(batch.max_seq_length):
            role_prob = self.role_controller.forward(batch)
            role_probs.append(role_prob)
            if t % self.args.role_change_interval == 0:
                role_indices = th.argmax(role_prob, dim = -1)
                self.role_embedding = self.role_embeddings[role_indices]
                
                role_llm_label, traj_thought_emb, traj_thought_str = self.llm_labeler(batch, t)
                role_llm_labels.append(role_llm_label)
                traj_thought_embs.append(traj_thought_emb)
                traj_thought_strs.append(traj_thought_str)

            agent_outs, traj_hidden_state = self.mac.forward(batch, t, self.role_embedding)
            mac_out.append(agent_outs)
            if t != 0 and t % self.args.role_change_interval == 0:
                traj_hidden_states.append(traj_hidden_state)
                traj_hid_align_out = self.traj_hid_alignment(traj_hidden_state)
                traj_hid_align_outs.append(traj_hid_align_out)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # TODO: add two losses
        # 1. role selector loss
        # 2. trajectory hidden alignment loss
        loss = L_td = masked_td_error.sum() / mask.sum()
        # TODO: calculate entory loss 
        loss_role_selector = self.criterion1(role_probs, role_llm_labels)
        # TODO: calculate the mse loss
        loss_traj_hid_alignment = self.criterion2(traj_hid_align_outs, traj_hidden_states)

        # TODO: add the optimizer process
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
