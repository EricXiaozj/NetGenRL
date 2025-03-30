# %%
from gensim.models import Word2Vec
import numpy as np
import json
import struct
import os

# %%
data_fold = '../data/iscx/'
bins_file = '../bins/bins_iscx.json'
IP_ATTRIBUTE_LIST = ['src_ip','dst_ip']
PORT_ATTRIBUTE_LIST = ['src_port','dst_port']
SERY_ATTRIBUTE_LIST = ['time','pkt_len','flags','ttl']
max_seq_len = 16
# NPRINT_LINE_LEN = 114

# %%
with open(bins_file, 'r') as f_bin:
    bins_data = json.load(f_bin)

# pkt_len_intervals = [bins['intervals'] for bins in bins_data['packet_len']]
# time_intervals = [bins['intervals'] for bins in bins_data['time']]
# pkt_len_intervals = bins_data['packet_len']['intervals']
# time_intervals = bins_data['time']['intervals']

# %%
def get_seqs_meta(data_fold):
    json_data_dic = {}
    for filename in os.listdir(data_fold):
        with open(data_fold + filename, 'r') as f:
            json_data = json.load(f)
            json_data_dic[filename.split('.')[0]] = json_data
        
    def find_interval(value, intervals):
        for idx, [start, end] in enumerate(intervals):
            if start <= value <= end:
                return idx  # 返回所在区间的下标
        return None
        
    seq_dic = {}
    
    
    for data in json_data_dic.values():
        for attribute in SERY_ATTRIBUTE_LIST:
            seq_dic[attribute] = []
            
    for idx, item in enumerate(data):
        for attribute in SERY_ATTRIBUTE_LIST:
            attr_seq = []
            for i, pkt in enumerate(item['series']):
                attr_seq.append(find_interval(pkt[attribute], bins_data[attribute]['intervals']))
                if i >= max_seq_len:
                    break
            seq_dic[attribute].append(attr_seq)
            
    meta_list = []
    for item in data:
        meta = []
        for attribute in IP_ATTRIBUTE_LIST + PORT_ATTRIBUTE_LIST:
            meta.append(item[attribute])
        meta_list.append(meta)
        
    return seq_dic, meta_list

# %%
def get_wv_model(seqs, vector_size, window, min_count = 1, sg = 1):
    seq_str = [[str(c) for c in seq] for seq in seqs]
    model = Word2Vec(sentences=seq_str, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def get_meta_model(seqs, vector_size, window, min_count = 1, sg = 1):
    seq_str = [[str(c)+f'_{i}' for i,c in enumerate(seq)] for seq in seqs]
    model = Word2Vec(sentences=seq_str, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

# %%
# 示例包长序列
sequences_dic, meta_list = get_seqs_meta(data_fold)

model_dic = {}

for key, seq in sequences_dic.items():
    model_dic[key] = get_wv_model(seq,8,5)
    model_dic[key].save(f"../wordvec/{key}.model")
    
meta_model = get_meta_model(meta_list,8,5)
meta_model.save(f"../wordvec/meta.model")
