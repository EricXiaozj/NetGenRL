import copy
import numpy as np
import torch

from discriminator import Discriminator

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = model.device

    def get_reward(self, x, num, discriminator, label, length): # num is the sample times of rollouts
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l, :]
                samples = self.own_model.sample(batch_size, label, length, data)
                # pred = discriminator(label, samples, length) # (batch_size)
                # pred_slice = pred[:,1].clone().view(-1)
                samples_wv = discriminator.seq2wv(samples)
                pred = discriminator.forward(label, samples_wv, length)
                # pred = discriminator(label, samples, length)
                pred_slice = pred.clone().view(-1)
                
                if i == 0:
                    rewards.append(pred_slice)
                else:
                    rewards[l-1] += pred_slice

            # for the last token
            # pred = discriminator(label, x, length)
            # pred_slice = pred.clone()[:,1].view(-1)
            samples_wv = discriminator.seq2wv(x)
            pred = discriminator.forward(label, samples_wv, length)
            pred_slice = pred.clone().view(-1)
            if i == 0:
                rewards.append(pred_slice)
            else:
                rewards[seq_len-1] += pred_slice
                
        rewards = torch.stack(rewards, dim=1) / (1.0 * num) 
        # print(rewards.shape)
        # rewards = torch.tensor(np.transpose(np.array(rewards))) / (1.0 * num) # (seq_len, batch_size) -> (batch_size, seq_len)
        mask = torch.zeros(rewards.shape).to(self.device)
        for l in range(seq_len):
            mask[:, l] = length.gt(l).float()
        rewards *= mask
        # print(length)
        # print(rewards)
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]