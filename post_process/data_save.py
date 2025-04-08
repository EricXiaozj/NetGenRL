import torch
from seqCGAN.generator import Generator  # 假设你有一个定义好的 Discriminator 类
# from seqCGAN.util import *
import json
# import random
# from torch.utils.data import DataLoader, Dataset
from post_process.check_bestmodel import get_real_data, get_fake_data


# label_dict = {'facebook': 0, 'skype': 1, 'email': 2, 'voipbuster': 3, 'youtube': 4, 'ftps': 5, 'vimeo': 6, 'spotify': 7, 'netflix': 8, 'bittorrent': 9}
# label_dict = {'benign_http': 0, 'benign_rtp': 1, 'bruteforce_dns': 2, 'bruteforce_ssh': 3, 'ddos_ack': 4, 'ddos_dns': 5, 'ddos_http': 6, 'ddos_syn': 7, 'scan_bruteforce': 8}
# label_dim = len(label_dict) 
# batch_size = 128
# dataset = 'ssrc_train'
# save_folder = './save_seq4/'
# # source_name = './data/vpn_data_small.json'
# # bins_name = './bins/bins_small_new.json'
# data_folder = './data/' + dataset + '/'
# bins_file_name = './bins/bins_' + dataset + '.json'
# wordvec_file_name = './wordvec/word_vec_' + dataset + '.json'
# save_folder_name = './data-generated/' + dataset + '/'
  
# def get_real_data(data_folder, label_dict, meta_attrs, sery_attrs, bins_data, max_seq_len):
#     data_dic = {}
#     for filename in os.listdir(data_folder):
#         with open(data_folder + filename, 'r') as f:
#             if filename.split('.')[0] not in label_dict.keys():
#                 continue
#             json_data = json.load(f)
#             data_dic[filename.split('.')[0]] = []
#             for item in json_data:
#                 meta_list = []
#                 for meta_attr in meta_attrs:
#                     meta_list.append(item[meta_attr]/bins_data[meta_attr]['intervals'][-1][1])
#                 count = 0
#                 seq = []
#                 for pkt in item['series']:
#                     attr_list = []
#                     for sery_attr in sery_attrs:
#                         attr_list.append(pkt[sery_attr]/bins_data[sery_attr]['intervals'][-1][1])
#                     seq.append(attr_list + meta_list)
#                     # seq.append(attr_list)
#                     count += 1
#                     if count >= max_seq_len:
#                         break
#                 data_dic[filename.split('.')[0]].append(seq)
#     return data_dic

# class SequenceDataset(Dataset):
#     def __init__(self, datas, label_str):
#         """
#         :param sequences: 一个包含真实序列的列表，每个序列是一个 ndarray 或 list
#         """
#         self.datas = datas
#         self.lengths = [len(seq) for seq in datas]  # 提取序列长度
#         label_id = label_dict[label_str]
#         self.label_one_hot = torch.zeros(label_dim).to('cpu') 
#         self.label_one_hot[label_id] = 1

#     def __len__(self):
#         return len(self.datas)

#     def __getitem__(self, idx):
#         return self.lengths[idx],self.label_one_hot
    
# def get_fake_data(real_data, label_str, generator):
#     dataset = SequenceDataset(real_data,label_str)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
#     generated_sequences = []
#     with torch.no_grad():
#         for lengths, labels in dataloader:
#             lengths = lengths.to(torch.device("cpu"))  # 确保在同一个设备上
#             labels = labels.to(torch.device("cpu"))
#             # print(lengths.shape)
#             # print(labels.shape)
#             batch_size = lengths.size(0)
#             # 生成随机噪声向量
#             # noise = torch.randn(len(lengths), noise_dim)
#             # 输入生成器生成数据
#             fake_data = generator.sample(batch_size,labels,lengths)
#             # 将生成结果按序列长度截断
#             for i, length in enumerate(lengths):
#                 generated_sequences.append(fake_data[i, :length].cpu().tolist())
       
#     final_seqs = []         
#     for seq in generated_sequences:
#         f_seq = []
#         for i in range(len(seq)):
#             pkt = []
#             for j,attr_id in enumerate(seq[i]): 
#                 if j == 0:
#                     attr = round(random.uniform(bins_data[SEQ_LIST[j]]['intervals'][attr_id][0], bins_data[SEQ_LIST[j]]['intervals'][attr_id][1]),2) 
#                 elif j < len(SERY_LIST) or i == 0:
#                     attr = round(random.uniform(bins_data[SEQ_LIST[j]]['intervals'][attr_id][0], bins_data[SEQ_LIST[j]]['intervals'][attr_id][1])) 
#                 else:
#                     attr = f_seq[0][j]
#                 # elif j < len(SERY_LIST):
#                 #    attr = round(random.uniform(bins_data[SEQ_LIST[j]]['intervals'][attr_id][0], bins_data[SEQ_LIST[j]]['intervals'][attr_id][1])) / bins_data[SEQ_LIST[j]]['intervals'][-1][1]
#                 # else:
#                 #    break
                
#                 pkt.append(attr)
#             f_seq.append(pkt)
#         final_seqs.append(f_seq)
#     return final_seqs 

# def pad(sequence, target_length, pad_value=np.nan):
#     seq_len = len(sequence)
#     if seq_len < target_length:
#         padding = [[pad_value] * len(sequence[0])] * (target_length - seq_len)
#         return sequence + padding  # 填充
#     return sequence

def save_data(label, data, meta_attrs, sery_attrs, result_folder_name):
    json_data = []
    for seq in data:
        item = {}
        for i, meta_attr in enumerate(meta_attrs):
            item[meta_attr] = seq[0][i+len(sery_attrs)]
        sery = []
        for pkt in seq:
            p = {}
            for i, sery_attr in enumerate(sery_attrs):
                p[sery_attr] = pkt[i]
            sery.append(p)
        item['series'] = sery
        json_data.append(item)
        
    with open(result_folder_name + label + '.json','w') as file:
        json.dump(json_data,file)
        
    print(f"Save data to {result_folder_name + label + '.json'}")
        

def generate_data(label_dict, dataset, json_folder, bins_folder, wordvec_folder, model_folder, result_folder, meta_attrs, sery_attrs, batch_size, max_seq_len, checkpoint, model_id, expand_times):
    label_dim = len(label_dict)
    save_folder = f'./{model_folder}/{dataset}/'
    data_folder = f'./{json_folder}/{dataset}/'
    bins_file_name = f'./{bins_folder}/bins_{dataset}.json'
    wordvec_file_name = f'./{wordvec_folder}/word_vec_{dataset}.json'
    result_folder_name = f'./{result_folder}/{dataset}/'
    seq_dim = len(meta_attrs) + len(sery_attrs)
    with open(wordvec_file_name, 'r') as f:
        wv_dict = json.load(f)
    
    wv = {}
    for key, metrics in wv_dict.items():
        wv[key] = torch.tensor(metrics, dtype=torch.float32)
    
    x_list = [wv_tensor.size(0) for wv_tensor in wv.values()]

    bins_data = {}
    with open(bins_file_name, 'r') as f_bin:
        bins_data = json.load(f_bin)

    real_datas = get_real_data(data_folder,label_dict,meta_attrs,sery_attrs,bins_data,max_seq_len)
    
    # model_name = save_folder + f'generator_{model_id}.pth'
    model_name = save_folder + f'generator_pre.pth'
        
    generator = Generator(label_dim,seq_dim,max_seq_len,x_list,'cpu')
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))  # 加载保存的权重字典
    generator.load_state_dict(checkpoint)  # 将权重字典加载到模型中
    generator.eval()
        
    fake_datas = {}
    # times = 20
    for label, data in real_datas.items():
        fake_data = get_fake_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
        fake_datas[label] = fake_data
        for _ in range(expand_times - 1):
            fake_datas[label] += get_fake_data(label_dict,label_dim,data,label,generator,bins_data,sery_attrs,meta_attrs,batch_size)
            
        print("Generate data of", label)
    
    for label, data in fake_datas.items():
        save_data(label,data,meta_attrs,sery_attrs,result_folder_name)
        
