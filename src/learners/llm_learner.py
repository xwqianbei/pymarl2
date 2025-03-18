import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.api import Call_API
from utils.text_embedding import TextEmbedding
import json

class LLMLearner:
    def __init__(self, mac, scheme, logger, args):
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

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

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

        # get llmer
        self.llmer = Call_API(args, map_name=getattr(args, "env_args", {}).get("map_name", "5m_vs_6m"))
        
        # 初始化技能到角色索引的映射
        self.skill_to_idx = {
            "Focus Fire": 0,
            "Retreat": 1,
            "Spread Out": 2,
            "Advance": 3,
            "Dead": 4
        }
        
        # 初始化文本嵌入模型
        self.text_embedder = None
        if hasattr(args, 'text_embedding_model_path'):
            self.text_embedder = TextEmbedding(args.text_embedding_model_path)
        
    def get_llm_output(self, batch, t):
        """return the role_label and role_thoughts of the agents
        Args:
            batch: the batch of episodes
            t: the time step
        Returns:
            role_label(torch.Tensor(bs, n_agents, role_num)): the role label of the agents
            role_thoughts(torch.Tensor(bs, n_agents, traj_embedding_dim)): the role thoughts of the agents
        """
        states = batch["state"][:, t] # [bs, state_dim]
        bs = states.shape[0]
        role_labels = []
        role_thoughts = []
        
        for state in states:
            # 调用LLM获取响应
            llm_response = self.llmer(state, self.args.n_agents)

            with open("llm_response.json", "w", encoding='utf-8') as f:
                json.dump(llm_response, f, indent=4, ensure_ascii=False)
            
            # 处理每个智能体的响应
            batch_agent_labels = []
            batch_agent_thoughts = []
            
            for agent_response in llm_response:
                # 获取技能并转换为one-hot编码
                skill = agent_response["skill"]
                skill_idx = self.skill_to_idx.get(skill, 0)  # 默认为Focus Fire
                
                one_hot = th.zeros(self.args.role_num)
                one_hot[skill_idx] = 1.0
                batch_agent_labels.append(one_hot)
                
                # 获取思考过程并转换为嵌入向量
                thought_process = agent_response["thought_process"]
                
                if self.text_embedder is not None:
                    # 使用文本嵌入模型获取嵌入向量
                    thought_embedding = self.text_embedder.embedding_text([thought_process])[0]
                else:
                    # 如果没有文本嵌入模型，则使用零向量
                    thought_embedding = th.zeros(self.args.traj_embedding_dim)
                
                batch_agent_thoughts.append(thought_embedding)
            
            # 将每个批次的智能体标签和思考添加到列表中
            role_labels.append(th.stack(batch_agent_labels))
            role_thoughts.append(th.stack(batch_agent_thoughts))
        
        # 将列表转换为张量
        role_labels = th.stack(role_labels)  # [bs, n_agents, role_num]
        role_thoughts = th.stack(role_thoughts)  # [bs, n_agents, traj_embedding_dim]
        
        return role_labels, role_thoughts

        
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
        change_turns = int(1 + batch.max_seq_length // self.args.role_change_interval)
        mac_out = []
        mac_out_role_probs = []# [max_seq_length, bs, n_agents, role_num]
        mac_out_traj_transfer_embd = []

        llm_out_role_labels = []
        llm_out_traj_thoughts = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs, role_probs, traj_transfer_embd = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            if t % self.args.role_change_interval == 0:
                mac_out_role_probs.append(role_probs)
                llm_role_labels, llm_role_thoughts = self.get_llm_output(batch, t)
                llm_out_role_labels.append(llm_role_labels)
                if t != 0:
                    mac_out_traj_transfer_embd.append(traj_transfer_embd)  
                llm_out_traj_thoughts.append(llm_role_thoughts) 
        
        # Concat over time
        mac_out = th.stack(mac_out, dim=1) # [bs, max_seq_length, n_agents, n_actions]
        mac_out_role_probs = th.stack(mac_out_role_probs, dim=1) # [bs, change_turns, n_agents, role_num]
        mac_out_traj_transfer_embd = th.stack(mac_out_traj_transfer_embd, dim=1) # [bs, change_turns - 1, n_agents, traj_embedding_dim]
        llm_out_role_labels = th.stack(llm_out_role_labels, dim=1) # [bs, change_turns, n_agents, role_num]
        llm_out_traj_thoughts = th.stack(llm_out_traj_thoughts, dim=1) # [bs, change_turns - 1, n_agents, traj_embedding_dim]

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

        L_td = masked_td_error.sum() / mask.sum()
        
        # 计算角色标签的交叉熵损失
        # mac_out_role_probs: [bs, change_turns, n_agents, role_num]
        # llm_out_role_labels: [bs, change_turns, n_agents, role_num]
        
        # 将张量移动到正确的设备
        mac_out_role_probs = mac_out_role_probs.to(self.device)
        llm_out_role_labels = llm_out_role_labels.to(self.device)
        
        # 重塑张量以适应交叉熵损失计算
        bs, change_turns, n_agents, role_num = mac_out_role_probs.shape
        mac_out_role_probs_flat = mac_out_role_probs.reshape(-1, role_num)  # [bs * change_turns * n_agents, role_num]
        llm_out_role_labels_flat = llm_out_role_labels.reshape(-1, role_num)  # [bs * change_turns * n_agents, role_num]
        
        # 获取llm_out_role_labels的最大索引作为目标类别
        llm_out_role_indices = th.argmax(llm_out_role_labels_flat, dim=1)  # [bs * change_turns * n_agents]
        
        # 计算交叉熵损失
        role_loss = F.cross_entropy(mac_out_role_probs_flat, llm_out_role_indices)
        L_role = role_loss * getattr(self.args, "role_loss_weight", 1.0)
        
        # 计算轨迹嵌入的余弦相似度损失
        # mac_out_traj_transfer_embd: [bs, change_turns - 1, n_agents, traj_embedding_dim]
        # llm_out_traj_thoughts: [bs, change_turns, n_agents, traj_embedding_dim]
        
        # 确保两个张量具有相同的形状
        if mac_out_traj_transfer_embd.shape[1] < llm_out_traj_thoughts.shape[1]:
            # 使用llm_out_traj_thoughts的前change_turns-1个时间步
            llm_out_traj_thoughts_matched = llm_out_traj_thoughts[:, 1:mac_out_traj_transfer_embd.shape[1]+1]
        else:
            llm_out_traj_thoughts_matched = llm_out_traj_thoughts[:, 1:]
        
        # 将张量移动到正确的设备
        mac_out_traj_transfer_embd = mac_out_traj_transfer_embd.to(self.device)
        llm_out_traj_thoughts_matched = llm_out_traj_thoughts_matched.to(self.device)
        
        # 重塑张量以计算余弦相似度
        mac_out_traj_flat = mac_out_traj_transfer_embd.reshape(-1, self.args.traj_embedding_dim)  # [bs * (change_turns-1) * n_agents, traj_embedding_dim]
        llm_out_traj_flat = llm_out_traj_thoughts_matched.reshape(-1, self.args.traj_embedding_dim)  # [bs * (change_turns-1) * n_agents, traj_embedding_dim]
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(mac_out_traj_flat, llm_out_traj_flat, dim=1)  # [bs * (change_turns-1) * n_agents]
        
        # 计算余弦相似度损失 (1 - 余弦相似度的平均值)
        traj_loss = 1.0 - cos_sim.mean()
        L_traj = traj_loss * getattr(self.args, "traj_loss_weight", 1.0)
        
        # 总损失
        loss = L_td + self.args.role_loss_weight * L_role + self.args.trajectory_loss_weight * L_traj

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
            self.logger.log_stat("loss_role", L_role.item(), t_env)
            self.logger.log_stat("loss_traj", L_traj.item(), t_env)
            self.logger.log_stat("loss_total", loss.item(), t_env)
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
