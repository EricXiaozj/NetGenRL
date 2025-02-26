# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import struct

# %%
NPRINT_LINE_LEN = 114
# NPRINT_REAL_WIDTH = 50*8
NPRINT_REAL_WIDTH = 22*8
# LABEL_DICT = {'facebook': 0, 'aim': 1, 'email': 2, 'voipbuster': 3, 'hangouts': 4, 'youtube': 5, 'sftp': 6, 'icq': 7, 'skype': 8, 'ftps': 9, 'vimeo': 10, 'spotify': 11, 'netflix': 12, 'bittorrent.json': 13}
LABEL_DICT = {'facebook': 0, 'skype': 1}
BYTES_LOG_MEAN = 5.961416185547372
BYTES_LOG_STD = 1.4073518072041993
PACKETS_LOG_MEAN = 1.5394048328338192
PACKETS_LOG_STD = 1.1067132728539024
# BYTES_MEAN = 82172.08261534268
# BYTES_STD = 5346671.4367792355
# PACKETS_MEAN = 78.89016816980528
# PACKETS_STD = 4953.4114679999075
# BYTES_LOG_MEAN = 6.175637510717154
# BYTES_LOG_STD = 1.529083592053985
# PACKETS_LOG_MEAN = 1.4661478151314664
# PACKETS_LOG_STD = 1.178370970139421
SEQ_DIM = 3
MAX_PKT_LEN = 1500
MAX_TIME = 1000
MAX_PORT = 65535
MAX_SEQ_LEN = 16
N_CRITIC = 5

# %%
class Generator(nn.Module):
    def __init__(self, label_dim, noise_dim, seq_dim, max_seq_len, device):
        super(Generator, self).__init__()
        
        self.label_dim = label_dim
        self.noise_dim = noise_dim
        # self.metadata_dim = metadata_dim
        self.hidden_dim = 512
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim
        self.device = device
        # self.output_dim = output_dim

        
        # MLP处理标签
        self.label_fc = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.ReLU(True),
        )
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 128),
            nn.ReLU(True),
        )
        
        # 噪声输入处理
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True)
        )

        self.combine_fc = nn.Sequential(
            nn.Linear(128+128+128,self.hidden_dim),
            nn.ReLU(True)
        )

        # self.metadata_out_fc = nn.Sequential(
        #     nn.Linear(128 + 128, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, metadata_dim),
        #     # nn.ReLU()  
        #     nn.Identity()
        # )

        # self.metadata_in_fc = nn.Sequential(
        #     nn.Linear(metadata_dim, 256),
        #     nn.ReLU(True),
        # )
        
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=4, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, seq_dim),
            # nn.Identity()
            nn.Tanh()
        )
        
    def forward(self, label, noise, length):
        
        # log_length = torch.log(length).to(self.device)
        # log_length = log_length.unsqueeze(1)
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # label_length = torch.cat([label,length_one_hot], dim=1)

        label_out = self.label_fc(label)
        noise_out = self.noise_fc(noise)
        length_out = self.length_fc(length_one_hot)

        combined = torch.cat([label_out, noise_out, length_out], dim=1)

        combined_out = self.combine_fc(combined)
        # 生成元数据
        # metadata_out = self.metadata_out_fc(combined1)

        # metadata_in = self.metadata_in_fc(metadata_out)
        # combined2 = torch.cat([label_out, noise_out, metadata_in], dim=1)

        x = combined_out.unsqueeze(1)

        output_seq = []

        for _ in range(self.max_seq_len):  # generate sequence with maximum length
            x, _ = self.lstm(x)  # LSTM output
            output = self.output_layer(x.squeeze(1))  # generate output at each time step
            output_seq.append(output)
        
        output_seq = torch.stack(output_seq, dim=1) 
        return output_seq


# %%
class Discriminator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, device):
        super(Discriminator, self).__init__()
        
        self.seq_dim = seq_dim
        # self.metadata_dim = metadata_dim
        self.label_dim = label_dim
        self.max_seq_len = max_seq_len
        self.device = device

        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=512, num_layers=4, batch_first=True)
        
        # MLP处理标签
        self.label_fc = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.ReLU(True),
        )
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 128),
            nn.ReLU(True),
        )
        
        # # MLP处理元数据
        # self.metadata_fc = nn.Sequential(
        #     nn.Linear(metadata_dim, 128),
        #     nn.ReLU(True),
        # )
        
        # 合并图像特征、标签特征和元数据特征
        self.fc = nn.Sequential(
            nn.Linear(128+128+512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Identity() # 输出真假
        )
        
    def forward(self, label, seq, length):
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # 处理标签
        label_out = self.label_fc(label)
        length_out = self.length_fc(length_one_hot)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True, enforce_sorted=False)

        # Process with LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # Unpack and get the last hidden state
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        last_hidden_state = output[range(len(output)), (length - 1).to(torch.long), :]  # Get the last valid hidden state

        combined = torch.cat([label_out,length_out,last_hidden_state], dim=1)

             
        # 判别真假
        validity = self.fc(combined)
        return validity



# %%
class CustomDataset(Dataset):
    def __init__(self, json_file, class_mapping, max_seq_len, bins_file, transform=None):
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
        # 将整数标签转换为 one-hot 编码
        labels_one_hot = F.one_hot(torch.tensor(label_int), num_classes=len(self.class_mapping)).float()

        port_intervals = self.bins_data['port']['intervals']
        pkt_len_intervals = []
        for bins in self.bins_data['packet_len']:
            pkt_len_intervals.append(bins['intervals'])
        time_intervals = []
        for bins in self.bins_data['time']:
            time_intervals.append(bins['intervals'])

        seq = []

        im = bytes.fromhex(item['nprint'])
        # def split_bytes_by_length(data, chunk_size):
        #     return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # lines = split_bytes_by_length(im, NPRINT_LINE_LEN)

        line = im[0:NPRINT_LINE_LEN]
        tcp_dport = line[32:34]
        udp_dport = line[92:94]
        dport = bytearray(a | b for a, b in zip(tcp_dport, udp_dport))
        dport = int.from_bytes(dport, 'big')
        
        
        dport_id = find_interval(dport,port_intervals)
        dport = dport_id/len(port_intervals)
        # dport /= MAX_PORT
        dport = dport * 2 - 1
        
        
        
        count = 0
        for i in range(0, len(im), NPRINT_LINE_LEN):
            # new_line = line[:22]+line[34:46]+line[98:]
            # line = bytes(line) 
            line = im[i:i+NPRINT_LINE_LEN]
            # print(line[0:8])
            time_h,time_l, pkt_len = struct.unpack("IIh", line[:10])
            time_l //= 1e4
            time = time_h + time_l/100
            
            time_id = find_interval(time,time_intervals[count])
            pkt_len_id = find_interval(pkt_len,pkt_len_intervals[count])
            # sign = -1
            
            # if pkt_len < 0:
            #     sign = 1
            #     pkt_len = -pkt_len
            
            time = time_id/len(time_intervals[count])
            pkt_len = pkt_len_id/len(pkt_len_intervals[count])
            
            time = time * 2 - 1
            pkt_len = pkt_len * 2 - 1
            
            seq.append([time,pkt_len,dport])
            count += 1
            if count >= self.max_seq_len:
                break
        
        # 填充 nprint，使其宽度固定
        if len(seq) < self.max_seq_len:
            seq = np.pad(seq, ((0, self.max_seq_len - len(seq)), (0, 0)), mode='constant', constant_values=0)
        
        
        # 转换为PyTorch的Tensor
        # log_metadata = torch.tensor(log_metadata, dtype=torch.float32)
        labels_one_hot = torch.tensor(labels_one_hot, dtype=torch.float32)  # 转换为Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        # print(seq)
        # length = np.abs(metadata[1])
        length = torch.tensor(length, dtype=torch.float32)
        
        return seq, labels_one_hot, length

# %%
class ConditionalGAN(nn.Module):
    def __init__(self, label_dim, noise_dim, seq_dim, max_seq_len, device):
        super(ConditionalGAN, self).__init__()
        # self.metaGenerator = MetaGenerator(label_dim,noise_dim,metadata_dim)
        self.generator = Generator(label_dim,noise_dim,seq_dim,max_seq_len,device)
        # self.metaDiscriminator = MetaDiscriminator(label_dim,metadata_dim)
        self.discriminator = Discriminator(label_dim,seq_dim,max_seq_len,device)
        
    def forward(self, label, noise, lengths):
        # 生成图像和元数据
        generated_seq, metadata = self.generator(label, noise, lengths)
        # lengths = torch.exp(metadata.detach()[:,1] * PACKETS_LOG_STD + PACKETS_LOG_MEAN)
        # 判别图像
        validity = self.discriminator(label, metadata,generated_seq,lengths)
        return generated_seq, validity


# %%
def get_non_matching_labels_one_hot(labels_indices, num_classes, device):
    # 获取当前批次的标签
    current_labels = labels_indices.clone()

    random_labels = torch.randint(0, num_classes, (labels_indices.size(0),), device=device)

    # 找到与 labels 中相同位置的索引
    same_positions = random_labels == current_labels

    # 对这些相同位置的数值进行修改
    while torch.any(same_positions):  # 如果有位置相同，继续修改
        random_labels[same_positions] = torch.randint(0, num_classes, (same_positions.sum(),), device=device)  # 修改相同位置的数值
        same_positions = random_labels == current_labels  # 更新相同位置
    
    # 将随机标签转换为 one-hot 编码
    wrong_labels_one_hot = torch.zeros(labels_indices.size(0), num_classes, device=device)
    wrong_labels_one_hot.scatter_(1, random_labels.view(-1, 1), 1)

    return wrong_labels_one_hot


def compute_gradient_penalty(discriminator, real_seqs, fake_seqs, labels, lengths, device,lambda_gp=10.0):
    """
    计算梯度惩罚以确保Lipschitz连续性。
    :param discriminator: 判别器模型
    :param real_images: 真实图像 batch
    :param fake_images: 生成图像 batch
    :param real_metadata: 真实图像对应的元数据 batch
    :param fake_metadata: 生成图像对应的元数据 batch
    :param device: 当前设备
    :return: 梯度惩罚值
    """
    batch_size = real_seqs.size(0)
    
    # 随机插值生成新的样本
    alpha = torch.rand(batch_size, 1, 1).to(device)  # 随机采样权重
    alpha = alpha.expand_as(real_seqs)
    interpolated_seqs = alpha * real_seqs + (1 - alpha) * fake_seqs
    interpolated_seqs.requires_grad_(True)
    # interpolated_metadata = alpha * real_metadata + (1 - alpha) * fake_metadata
    
    # 将插值图像和元数据合并
    # interpolated_input = (interpolated_images, interpolated_metadata)

    # 计算插值样本的判别器输出
    interpolated_scores = discriminator(labels,interpolated_seqs,lengths)  # 注意这里要传入图像和元数据

    # 计算梯度
    gradients = torch.autograd.grad(outputs=interpolated_scores, inputs=interpolated_seqs,
                                   grad_outputs=torch.ones_like(interpolated_scores),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # 计算梯度的L2范数
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # 计算梯度惩罚
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # L2范数距离1的平方
    return gradient_penalty*lambda_gp

# %%
def train(conditionGAN, dataloader, epochs, device, clip_value=0.01, alpha = 1.0):
    # 使用Wasserstein GAN损失
    # optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = optim.RMSprop(conditionGAN.generator.parameters(), lr=0.00005)
    # optimizer_md = optim.RMSprop(conditionGAN.metaDiscriminator.parameters(), lr=0.0002)
    optimizer_d = optim.RMSprop(conditionGAN.discriminator.parameters(), lr=0.00005)

    for epoch in range(epochs):
        for i, (seqs, labels, lengths) in enumerate(dataloader):
            batch_size = seqs.size(0)
            seqs = seqs.to(device)
            labels = labels.to(device)
            # metadatas = metadatas.to(device)


            # 创建标签
            # valid = torch.ones(batch_size, 1).to(device)   # 真实数据+对应标签 -> 1
            # fake = torch.zeros(batch_size, 1).to(device)   # 生成数据+标签 -> 0

            # 训练判别器
            # optimizer_md.zero_grad()
            for _ in range(N_CRITIC):
                optimizer_d.zero_grad()
                real_validity = conditionGAN.discriminator(labels, seqs, lengths)
                noise = torch.randn(batch_size, conditionGAN.generator.noise_dim).to(device)
                fake_seqs = conditionGAN.generator(labels, noise,lengths)
                fake_validity = conditionGAN.discriminator(labels, fake_seqs.detach(),lengths)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) 
            
                d_loss_real = -torch.mean(real_validity)
                d_loss_fake = torch.mean(fake_validity)
                
                with torch.backends.cudnn.flags(enabled=False):
                    gp = compute_gradient_penalty(conditionGAN.discriminator, seqs, fake_seqs.detach(),labels, lengths,device)
                total_d_loss = d_loss + gp
                total_d_loss.backward()
                optimizer_d.step()
                

                # for p in conditionGAN.discriminator.parameters():
                #     p.data.clamp_(-clip_value, clip_value)

            # 训练生成器
            optimizer_g.zero_grad()

            # 生成器试图欺骗判别器（生成器希望判别器输出1）
            validity = conditionGAN.discriminator(labels, fake_seqs,lengths)
            g_loss = -torch.mean(validity)

            # 反向传播并更新生成器
            g_loss.backward()
            optimizer_g.step()

        # 每个epoch结束时保存模型
        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()} ({d_loss_real.item()}, {d_loss_fake.item()}), G Loss: {g_loss.item()}.")
        # print(f"Epoch [{epoch+1}/{epochs}], Real: {rp}, Fake: {fp}.")

        torch.save(conditionGAN.generator.state_dict(), './save_new/generator.pth')
        torch.save(conditionGAN.discriminator.state_dict(), './save_new/discriminator.pth')

        if (epoch + 1) % 50 == 0:
            torch.save(conditionGAN.generator.state_dict(), f'./save_new/generator_{epoch+1}.pth')
            torch.save(conditionGAN.discriminator.state_dict(), f'./save_new/discriminator_{epoch+1}.pth')


# %%
def generate_data(generator, label, noise, device):
    # 设置模型为评估模式
    generator.eval()

    # 生成图像和元数据
    generated_images, metadata = generator(label, noise)
    
    return generated_images, metadata

# %%
if __name__ == '__main__':
    label_dim = len(LABEL_DICT) 
    # image_dim = (1, NPRINT_REAL_WIDTH, NPRINT_REAL_WIDTH)  # 生成单通道图像
    # metadata_dim = 2  # 元数据维度为2
    noise_dim = 100  # 噪声维度
    batch_size = 64
    epochs = 1000
    seq_dim = SEQ_DIM
    max_seq_len = MAX_SEQ_LEN

    # 创建生成器、判别器和cGAN
    # generator = Generator(label_dim, noise_dim, image_dim, metadata_dim)
    # discriminator = Discriminator(label_dim, image_dim, metadata_dim)
    # cgan = ConditionalGAN(generator, discriminator)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    cg = ConditionalGAN(label_dim,noise_dim,seq_dim,max_seq_len,device)
    
    cg.to(device)

    print(device)

    print("Process dataset...")

    dataset = CustomDataset(json_file='./vpn_data_small.json', class_mapping=LABEL_DICT,max_seq_len=max_seq_len,bins_file= './bins.json')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # checkpoint_g = torch.load('./save_new/generator_v1.pth')  # 加载保存的权重字典
    # cg.generator.load_state_dict(checkpoint_g) 

    # checkpoint_d = torch.load('./save_new/discriminator_v1.pth')  # 加载保存的权重字典
    # cg.discriminator.load_state_dict(checkpoint_d) 

    print("Trainning...")

    # 训练模型（假设有dataloader加载数据）
    # 训练时需要提供真实数据：real_images, labels, metadata
    train(cg, dataloader, epochs, device)


    # 假设你有`labels`，创建噪声并生成数据
    # labels = torch.zeros(batch_size, label_dim).to(device)  # 初始化为全 0
    # labels[:, 0] = 1  # 将每个样本的第一个值设置为 1，形成 one-hot 编码    
    # noise = torch.randn(batch_size, noise_dim).to(device)    # 使用训练后的模型生成数据    
    # generated_images, metadata = generate_data(cg.generator, labels, noise, device)

    # print(metadata)
    # print(generated_images)


