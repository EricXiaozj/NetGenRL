import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import struct
from util import *

class CustomDataset(Dataset):
    def __init__(self, json_file, bins_file, class_mapping, max_seq_len, transform=None):
        """
        :param json_file: 存储数据的JSON文件路径
        :param class_mapping: 类别名到整数标签的映射
        :param nprint_width: nprint的固定宽度
        :param transform: 图像预处理转换
        """
        self.json_file = json_file
        self.class_mapping = class_mapping  # 类别映射
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # 读取JSON文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']
            
        with open(bins_file, 'r') as f_bin:
            self.bins_data = json.load(f_bin)
        
        # with open(set_file, 'r') as f_set:
        #     self.set_data = json.load(f_set)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        def find_interval(value, intervals):
            for idx, [start, end] in enumerate(intervals):
                if start <= value <= end:
                    return idx  # 返回所在区间的下标
            return None
        # 解析每一条数据
        item = self.data[idx]

        metadata = np.array(list(item['meta'].values()), dtype=np.float32)
        length = min(metadata[1],self.max_seq_len)
        # log_metadata = np.log1p(metadata)
        # log_metadata[0] = (log_metadata[0]-BYTES_LOG_MEAN)/BYTES_LOG_STD
        # log_metadata[1] = (log_metadata[1]-PACKETS_LOG_MEAN)/PACKETS_LOG_STD


        label_str = item['labels'][0]  # 假设 labels 是字符串类型
        # 将字符串标签转换为整数
        label_int = self.class_mapping[label_str]  # 使用类别映射字典
        weight = LABEL_COUNT_DICT[label_str] * len(LABEL_COUNT_DICT)
        # 将整数标签转换为 one-hot 编码
        labels_one_hot = F.one_hot(torch.tensor(label_int), num_classes=len(self.class_mapping)).float()

        # port_intervals = self.bins_data['port']['intervals']
        # pkt_len_intervals = []
        # for bins in self.bins_data['packet_len']:
        #     pkt_len_intervals.append(bins['intervals'])
        # time_intervals = []
        # for bins in self.bins_data['time']:
        #     time_intervals.append(bins['intervals'])
        
        pkt_len_intervals = self.bins_data['packet_len']['intervals']
        time_intervals = self.bins_data['time']['intervals']

        seq = []

        im = bytes.fromhex(item['nprint'])
        # def split_bytes_by_length(data, chunk_size):
        #     return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # lines = split_bytes_by_length(im, NPRINT_LINE_LEN)

        line = im[0:NPRINT_LINE_LEN]
        # tcp_dport = line[32:34]
        # udp_dport = line[92:94]
        # dport = bytearray(a | b for a, b in zip(tcp_dport, udp_dport))
        # dport = int.from_bytes(dport, 'big')
        
        
        # dport_id = find_interval(dport,port_intervals)
        # dport = dport_id/len(port_intervals)
        # # dport /= MAX_PORT
        # dport = dport * 2 - 1
        
        
        
        count = 0
        for i in range(0, len(im), NPRINT_LINE_LEN):
            # new_line = line[:22]+line[34:46]+line[98:]
            # line = bytes(line) 
            line = im[i:i+NPRINT_LINE_LEN]
            # print(line[0:8])
            time_h,time_l, pkt_len = struct.unpack("IIh", line[:10])
            time_l //= 1e4
            time = time_h + time_l / 100
            # time = (time_h % 1000) * 10 + time_l
            # pkt_len += 1500
            
            time_id = find_interval(time,time_intervals)
            pkt_len_id = find_interval(pkt_len,pkt_len_intervals)
            # sign = -1
            
            # if pkt_len < 0:
            #     sign = 1
            #     pkt_len = -pkt_len
            
            # time = time_id/len(time_intervals[count])
            # pkt_len = pkt_len_id/len(pkt_len_intervals[count])
            
            # time = time * 2 - 1
            # pkt_len = pkt_len * 2 - 1
            
            # seq.append([time,pkt_len,dport])
            seq.append([time_id,pkt_len_id])
            count += 1
            if count >= self.max_seq_len:
                break
        
        # 填充 nprint，使其宽度固定
        if len(seq) < self.max_seq_len:
            seq = np.pad(seq, ((0, self.max_seq_len - len(seq)), (0, 0)), mode='constant', constant_values=0)
        
        
        # 转换为PyTorch的Tensor
        # log_metadata = torch.tensor(log_metadata, dtype=torch.float32)
        # labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)  # 转换为Tensor
        seq = torch.tensor(seq, dtype=torch.long)
        # print(seq)
        # length = np.abs(metadata[1])
        length = torch.tensor(length, dtype=torch.float32)
        weight = torch.tensor(weight, dtype=torch.float32)
        
        return seq, labels_one_hot, length, weight
