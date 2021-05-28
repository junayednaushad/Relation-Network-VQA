import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class QuestionEmbedModel(nn.Module):
    def __init__(self, in_size, embed=32, hidden=256):
        super(QuestionEmbedModel, self).__init__()
        
        self.wembedding = nn.Embedding(in_size + 1, embed)  #word embeddings have size 32
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)  # Input dim is 32, output dim is the question embedding
        self.hidden = hidden
        
    def forward(self, question):
        #calculate question embeddings
        wembed = self.wembedding(question)
        # wembed = wembed.permute(1,0,2) # in lstm minibatches are in the 2-nd dimension
        self.lstm.flatten_parameters()
        _, hidden = self.lstm(wembed) # initial state is set to zeros by default
        qst_emb = hidden[0] # hidden state of the lstm. qst = (B x 256)
        #qst_emb = qst_emb.permute(1,0,2).contiguous()
        #qst_emb = qst_emb.view(-1, self.hidden*2)
        qst_emb = qst_emb[0]
        
        return qst_emb

class RelationalLayerBase(nn.Module):
    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__()

        self.f_fc1 = nn.Linear(hyp["g_layers"][-1], hyp["f_fc1"])
        self.f_fc2 = nn.Linear(hyp["f_fc1"], hyp["f_fc2"])
        self.f_fc3 = nn.Linear(hyp["f_fc2"], out_size)
    
        self.dropout = nn.Dropout(p=hyp["dropout"])
        
        self.on_gpu = False
        self.hyp = hyp
        self.qst_size = qst_size
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self):
        self.on_gpu = True
        super().cuda()

class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, qst_size, hyp):
        super().__init__(in_size, out_size, qst_size, hyp)

        self.quest_inject_position = hyp["question_injection_position"]
        self.in_size = in_size              # in_size is the size of the object pair vector (7+7)

	    #create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]
        for idx,g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx==0 else hyp["g_layers"][idx-1]
            out_s = g_layer_size
            if idx==self.quest_inject_position:
                layer = nn.Linear(in_s+qst_size, out_s)
            else:
                layer = nn.Linear(in_s, out_s)
            self.g_layers.append(layer)	
        self.g_layers = nn.ModuleList(self.g_layers)
    
    def forward(self, x, qst):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()
        qst_size = qst.size()[1]
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)                      # (B x 1 x 128)
        qst = qst.repeat(1, d, 1)                       # (B x 64 x 128)
        qst = torch.unsqueeze(qst, 2)                      # (B x 64 x 1 x 128)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)                   # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d, 1, 1)                    # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x, 2)                   # (B x 64 x 1 x 26)
        #x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d, 1)                    # (B x 64 x 64 x 26)
        
        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)                  # (B x 64 x 64 x 2*26)
        
        # reshape for passing through network
        x_ = x_full.view(b * d**2, self.in_size)

        #create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            if idx==self.quest_inject_position:
                in_size = self.in_size if idx==0 else self.g_layers_size[idx-1]

                # questions inserted
                x_img = x_.view(b,d,d,in_size)
                qst = qst.repeat(1,1,d,1)
                x_concat = torch.cat([x_img,qst],3) #(B x 64 x 64 x 128+256)

                # h layer
                x_ = x_concat.view(b*(d**2),in_size+self.qst_size)
                x_ = g_layer(x_)
                x_ = F.relu(x_)
            else:
                x_ = g_layer(x_)
                x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(b, d**2, self.g_layers_size[-1])
        x_g = x_g.sum(1).squeeze(1)
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = self.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f
        # return F.log_softmax(x_f, dim=1)

class RN(nn.Module):
    def __init__(self, args, hyp):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False    
            
        # LSTM
        hidden_size = hyp["lstm_hidden"]
        self.text = QuestionEmbedModel(args.qdict_size, embed=hyp["lstm_word_emb"], hidden=hidden_size)
        
        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl_out_size = args.adict_size
        self.rl = RelationalLayer(self.rl_in_size, self.rl_out_size, hidden_size, hyp) 
        if hyp["question_injection_position"] != 0:          
            print('Supposing IR model')
        else:     
            print('Supposing original DeepMind model')

    def forward(self, img, qst_idxs):
        x = img # (B x 12 x 8)  
        qst = self.text(qst_idxs)
        y = self.rl(x, qst)
        return y
    
    def cuda(self):
        self.on_gpu = True
        self.rl.cuda()
        super(RN, self).cuda()