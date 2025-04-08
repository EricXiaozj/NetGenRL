import torch
from seqCGAN.generator import Generator  # 假设你有一个定义好的 Discriminator 类
import json
from post_process.check_bestmodel import get_real_data, get_fake_data

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
        
