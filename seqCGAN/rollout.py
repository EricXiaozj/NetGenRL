import copy
import numpy as np
import torch
import time

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
        sample_time = 0
        seq2wv_time = 0
        forward_time = 0
        # for i in range(num):
        #     for l in range(1, seq_len):
        #         # data = x[:, 0:l, :]
                
        #         start_time = time.perf_counter()
        #         samples = self.own_model.sample(batch_size, label, length, x[:, :l, :])
        #         sample_time += time.perf_counter() - start_time
                
        #         # pred = discriminator(label, samples, length) # (batch_size)
        #         # pred_slice = pred[:,1].clone().view(-1)
                
        #         start_time = time.perf_counter()
        #         samples_wv = discriminator.seq2wv(samples)
        #         seq2wv_time += time.perf_counter() - start_time
                
        #         start_time = time.perf_counter()
        #         pred = discriminator.forward(label, samples_wv, length)
        #         forward_time += time.perf_counter() - start_time
        #         # pred = discriminator(label, samples, length)
        #         pred_slice = pred.clone().view(-1)
                
        #         if i == 0:
        #             rewards.append(pred_slice)
        #         else:
        #             rewards[l-1] += pred_slice

        #     # for the last token
        #     # pred = discriminator(label, x, length)
        #     # pred_slice = pred.clone()[:,1].view(-1)
        #     samples_wv = discriminator.seq2wv(x)
        #     pred = discriminator.forward(label, samples_wv, length)
        #     pred_slice = pred.clone().view(-1)
        #     if i == 0:
        #         rewards.append(pred_slice)
        #     else:
        #         rewards[seq_len-1] += pred_slice
        
        num = 16
        for l in range(1, seq_len):
                # data = x[:, 0:l, :]
            labels = label.repeat(num, 1).to(self.device)
            lengths = length.repeat(num)
            xs = x.repeat(num, 1, 1).to(self.device)
            
            # print(labels.shape, lengths.shape, xs.shape)
            # print(label.device, length.device, x.device)
            
            start_time = time.perf_counter()
            samples = self.own_model.sample(batch_size * num, labels, lengths, xs[:, :l, :])
            sample_time += time.perf_counter() - start_time
                
                # pred = discriminator(label, samples, length) # (batch_size)
                # pred_slice = pred[:,1].clone().view(-1)
            # for i in range(num):
            start_time = time.perf_counter()
            # samples_wv = discriminator.seq2wv(samples[i*batch_size:(i+1)*batch_size,:,:])
            samples_wv = discriminator.seq2wv(samples)
            seq2wv_time += time.perf_counter() - start_time
                
            start_time = time.perf_counter()
            pred = discriminator.forward(labels, samples_wv, lengths)
            forward_time += time.perf_counter() - start_time
            # pred = discriminator(label, samples, length)
            pred_slice = pred.clone().view(-1)
            
            #print(pred_slice.shape)
            
            for i in range(num):
                if i == 0:
                    rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
                else:
                    rewards[l-1] += pred_slice[i*batch_size:(i+1)*batch_size]
                
                # if i == 0:
                #     rewards.append(pred_slice)
                # else:
                #     rewards[l-1] += pred_slice

            # for the last token
            # pred = discriminator(label, x, length)
            # pred_slice = pred.clone()[:,1].view(-1)
            samples_wv = discriminator.seq2wv(xs)
            pred = discriminator.forward(labels, samples_wv, lengths)
            pred_slice = pred.clone().view(-1)
            
        for i in range(num):
            if i == 0:
                rewards.append(pred_slice[i*batch_size:(i+1)*batch_size])
            else:
                rewards[seq_len-1] += pred_slice[i*batch_size:(i+1)*batch_size]
            # if i == 0:
            #     rewards.append(pred_slice)
            # else:
            #     rewards[seq_len-1] += pred_slice
                
        # print('sample_time: ', sample_time, 'seq2wv_time: ', seq2wv_time, 'forward_time: ', forward_time)
                
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