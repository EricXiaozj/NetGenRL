import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, label_dim, seq_dim, max_seq_len, x_list, device):
        super(Generator, self).__init__()
        
        self.label_dim = label_dim
        # self.noise_dim = noise_dim
        self.hidden_dim = 512
        self.lstm_layers = 4
        self.embedding_dim = 128
        self.attribute_dim = 64
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim
        self.pred_dim = sum(x_list)
        self.x_list = x_list # list of seq value dim
        self.device = device
        self.count_list = []
        # count = 0
        # for x_len in x_list:
        #     count += x_len
        #     self.count_list.append(count)
                
        self.emb = nn.ModuleList([
            nn.Embedding(x_len, self.embedding_dim) for x_len in x_list
        ]) 
        
        # self.label_fc = nn.Sequential(
        #     nn.Linear(label_dim, 64),
        #     nn.ReLU(True),
        # )
        
        self.emb_now = nn.ModuleList([
            nn.Embedding(x_len, self.embedding_dim) for x_len in x_list
        ]) 
        
        self.condition_fix = nn.ModuleList([
            nn.Sequential(
                nn.Linear((i + 1)*self.embedding_dim, self.attribute_dim),
                nn.ReLU(True),
            ) for i in range(len(x_list))
        ])
        
        self.length_fc = nn.Sequential(
            nn.Linear(max_seq_len, 64),
            nn.ReLU(True),
        ) 
        
        # self.combine_fc = nn.Sequential(
        #     nn.Linear(128*len(x_list)+64+64,self.hidden_dim),
        #     nn.ReLU(True)
        # )
        
        self.combine_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128*len(x_list)+64,self.hidden_dim),
                nn.ReLU(True)
            ) for _ in range(self.label_dim)
        ])
        
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        self.lstm.flatten_parameters()

        self.output_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + self.attribute_dim, x_len),
                nn.Identity()
            ) for x_len in x_list
        ])
        
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def init_hidden(self, batch_size):
        h = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        c = torch.zeros((self.lstm_layers, batch_size, self.hidden_dim),device=self.device)
        return h, c
        
    def forward(self, label, length, x, x_now): # x: (batch_size, seq_len, seq_dim)
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # label_out = self.label_fc(label)
        label_int = torch.argmax(label.clone(),1)
        # length_out = torch.stack([self.length_fc[idx](length_one_hot[i]) for i, idx in enumerate(label_int)])
        length_out = self.length_fc(length_one_hot)
        
        # emb_list = torch.stack([torch.cat([self.emb[idx][j](x[i,:,j]) for j in range(len(self.x_list))], dim=1) for i, idx in enumerate(label_int)],dim=0)
        emb_list = torch.cat([self.emb[i](x[:,:,i]) for i in range(len(self.x_list))], dim=2)
        
        now_emb_list = torch.cat([self.emb_now[i](x_now[:,:,i]) for i in range(len(self.x_list))], dim=2)
        attribute_features = [self.condition_fix[i](now_emb_list[:,:,:(i+1)*self.embedding_dim]) for i in range(len(self.x_list))]
        attribute_features = [torch.zeros(x_now.shape[0],x_now.shape[1],self.attribute_dim).to(self.device)] + attribute_features[:-1]
        
        # label_expand = label_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)
        length_expand = length_out.unsqueeze(1).expand(-1, emb_list.size(1), -1)

        combined = torch.cat([emb_list, length_expand], dim=2)
        # print(combined.shape)
        # combined_out = self.combine_fc(combined)   
        # combined_out = torch.stack([self.combine_fc[idx](combined[i,:,:]) for i, idx in enumerate(label_int)],dim = 0)
        
        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        # print(fc_outputs.shape)
        
        indices = label_int.view(-1, 1, 1, 1).expand(-1, -1, fc_outputs.size(2), fc_outputs.size(3))
        # print(indices.shape)

        combined_out = torch.gather(fc_outputs, dim=1, index=indices).squeeze(1)

        # print(combined_out.shape)
        h0, c0 = self.init_hidden(combined_out.size(0))
        hidden_output, (h, c) = self.lstm(combined_out, (h0, c0))
        
        
        # output = self.output_layer(hidden_output.contiguous().view(-1, self.hidden_dim))
        
        # count = 0
        preds = []
        for i in range(len(self.x_list)):
            hidden_w_attr = torch.cat([hidden_output.clone().to(self.device), attribute_features[i]], dim=2)
            output = self.output_layer[i](hidden_w_attr.contiguous().view(-1, self.hidden_dim + self.attribute_dim))
            # pred = self.softmax(output[:,count:count + x_len])
            pred = F.softmax(output,dim=-1)
            preds.append(pred)
            # count += x_len
            
        preds = torch.cat(preds, dim=1).view(-1, self.max_seq_len, self.pred_dim) 
        return preds
    
    def step(self, label, length, x, x_now, dim_now, h, c):
        """
        Args:
            x: (batch_size, seq_dim), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        
        length_one_hot = F.one_hot((length.clone().detach().long() - 1), num_classes=self.max_seq_len).float().to(self.device)
        
        # label_out = self.label_fc(label)
        length_out = self.length_fc(length_one_hot)
        label_int = torch.argmax(label.clone(),1)
        # length_out = torch.stack([self.length_fc[idx](length_one_hot[i]) for i, idx in enumerate(label_int)])
        
        # print(x.shape)
        emb_list = torch.cat([self.emb[i](x[:,i]) for i in range(len(self.x_list))], dim=1)
        
        if dim_now > 0 and x_now is not None:
            now_emb_list = torch.cat([self.emb_now[i](x_now[:,i]) for i in range(dim_now)], dim=1)
            attribute_feature = self.condition_fix[dim_now - 1](now_emb_list)
        else:
            attribute_feature = torch.zeros(x.shape[0],self.attribute_dim).to(self.device)

        # combined = torch.cat([emb_list, label_out, length_out], dim=1)
        # combined_out = self.combine_fc(combined)   
        combined = torch.cat([emb_list, length_out], dim=1)
        # combined_out = torch.stack([self.combine_fc[idx](combined[i]) for i, idx in enumerate(label_int)])

        fc_outputs = torch.stack([self.combine_fc[idx](combined) for idx in range(len(self.combine_fc))], dim=1)
        
        indices = label_int.view(-1, 1, 1).expand(-1, -1, fc_outputs.size(2))

        combined_out = torch.gather(fc_outputs, dim=1, index=indices)

        hidden_output, (h, c) = self.lstm(combined_out, (h, c))
        
        hidden_output = hidden_output.squeeze(1)
        # print(hidden_output.shape, attribute_feature.shape)
        hidden_w_attr = torch.cat([hidden_output.clone(), attribute_feature], dim=1)
        output = self.output_layer[dim_now](hidden_w_attr.contiguous())
        pred = F.softmax(output,dim=-1)
        
        # output = self.output_layer(hidden_output.view(-1, self.hidden_dim))
        
        # count = 0
        # preds = []
        # for x_len in self.x_list:
        #     pred = F.softmax(output[:,count:count + x_len],dim=-1)
        #     preds.append(pred)
        #     count += x_len
            
        # preds = torch.cat(preds, dim=1) 
        
        
            
        return pred, h, c
    
    
    def sample(self, batch_size, label, length, x=None):
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = torch.zeros((batch_size, self.seq_dim)).long().to(self.device)
            
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(self.max_seq_len): 
                last = None
                sam = []
                for j in range(len(self.x_list)):
                    pred, h, c = self.step(label, length, x, last, j, h, c)
                    sam.append(pred.multinomial(1))
                    last = torch.cat(sam,dim=1)
                x = last
                # preds, h, c = self.step(label, length, x, h, c)
                # sam = []
                # count = 0
                # for x_len in self.x_list:
                #     pred = preds[:,count:count + x_len]
                #     # print(pred)
                #     sam.append(pred.multinomial(1))
                #     count += x_len
                # x = torch.cat(sam,dim=1)
                samples.append(x)
        else:
            # print(x.shape)
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            sam = []
            for i in range(given_len):
                input = lis[i].squeeze(1)
                for j in range(len(self.x_list)):
                    if j == 0:
                        pred, h, c = self.step(label, length, input, None, j, h, c)
                    else:
                        pred, h, c = self.step(label, length, input, input[:,:j], j, h, c)
                    if i == given_len - 1:
                        sam.append(pred.multinomial(1))
                # preds, h, c = self.step(label, length, input, h, c)
                samples.append(input)
                # print(input.shape)
            
            # count = 0
            # for x_len in self.x_list:
            #     pred = preds[:,count:count + x_len]
            #     sam.append(pred.multinomial(1))
            #     count += x_len
            sam_tensor = torch.cat(sam,dim=1)
            # print(sam_tensor.shape)
            
            for i in range(given_len, self.max_seq_len):
                samples.append(sam_tensor)
                sam = []
                last = None
                for j in range(len(self.x_list)):
                    pred, h, c = self.step(label, length, sam_tensor, last, j, h, c)
                    sam.append(pred.multinomial(1))
                    last = torch.cat(sam,dim=1)
                sam_tensor = last
                    # preds, h, c = self.step(label, length, sam_tensor, h, c)
                # count = 0
                # sam = []
                # for x_len in self.x_list:
                #     pred = preds[:,count:count + x_len]
                #     # print(pred)
                #     sam.append(pred.multinomial(1))
                #     count += x_len
                # sam_tensor = torch.cat(sam,dim=1)
                # print(sam_tensor.shape)
        output = torch.stack(samples, dim=1)
        # print(output.shape)
        return output