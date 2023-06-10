import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

class Embedding_Layer(nn.Module):
    # feature embedding layer
    def __init__(self):
        super(Embedding_Layer, self).__init__()
        self.feat_emb = nn.ModuleList([nn.Embedding(feature_size, 8) for feature_size in [2405, 24, 7]])
        for ele in self.feat_emb:
            nn.init.xavier_uniform_(ele.weight.data, gain=math.sqrt(2.0))
        
    def forward(self, X):
        B, N, T, F = X.size() # (B, N, T, F)
        X_feat =torch.cat([emb(X[:, :, :, i+3].long()) for i, emb in enumerate(self.feat_emb)], dim=-1)
        return X_feat

def apply_bn(x):
    # batch normalization of 3D tensor x
    bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
    x = bn_module(x)
    return x

class GCN_2D(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hop=1):
        super(GCN_2D, self).__init__()
        self.in_features = in_features
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.w_lot = nn.ModuleList()
        for i in range(hop):
            in_features = (self.in_features) if(i==0) else out_features 
            self.w_lot.append(nn.Linear(in_features, out_features, bias=True))

    def forward(self, x, adj):
        # node aggregation
        for i in range(self.hop):
            x = torch.mm(adj.float(), x.float())
            x = self.leakyrelu(self.w_lot[i](x)) #(B, N, F)
        return x

class GCN(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hop=1):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.w_lot = nn.ModuleList()
        for i in range(hop):
            in_features = (self.in_features) if(i==0) else out_features 
            self.w_lot.append(nn.Linear(in_features, out_features, bias=True))

    def forward(self, x, adj):
        # node aggregation
        for i in range(self.hop):
            x = torch.einsum('bwf,vw->bvf', (x, adj.float()))
            x = self.leakyrelu(self.w_lot[i](x)) #(B, N, F)
        return x
    
class N_Loss(nn.Module):
    def __init__(self):
        super(N_Loss, self).__init__()
        self.positive_num = 0
        self.negative_num = 0
        self.positive_pairs = []
        self.negative_pairs = []
        
    def get_loss(self, x_embed, nhid):
        pos_embed1 = x_embed[:, self.positive_pairs[:, 0], :].view(-1, nhid)
        pos_embed2 = x_embed[:, self.positive_pairs[:, 1], :].view(-1, nhid)
        pos_score = F.cosine_similarity(pos_embed1, pos_embed2)
        neg_embed1 = x_embed[:, self.negative_pairs[:, 0], :].view(-1, nhid)
        neg_embed2 = x_embed[:, self.negative_pairs[:, 1], :].view(-1, nhid)
        neg_score = F.cosine_similarity(neg_embed1, neg_embed2)
        
        neg_score = 4*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
        pos_score = torch.mean(torch.log(torch.sigmoid(pos_score)), 0)
        node_score = -pos_score-neg_score
        
        return node_score

    def get_positive_node_pair(self, adj, num):
        self.positive_num = num
        positive_indices = torch.nonzero(adj > 0.2)
        random_indices = torch.randperm(positive_indices.size(0))[:num]
        self.positive_pairs = positive_indices[random_indices, :]
        
        return self.positive_pairs
    
    def get_negative_node_pair(self, adj, num):
        self.negative_num = num
        negative_indices = torch.nonzero(adj <= 0.2)
        random_indices = torch.randperm(negative_indices.size(0))[:num]
        self.negative_pairs = negative_indices[random_indices, :]
        
        return self.negative_pairs

class SSL_GNN(nn.Module):
    def __init__(self, device, idx_label, idx_unlabel, nhid, dropout, alpha=0.2):
        super(SSL_GNN, self).__init__()
        self.device = device
        self.idx_label = idx_label
        self.idx_unlabel = idx_unlabel
        self.nhid = nhid
        
        # embed_layer = Embedding_Layer()

        # Approximation
        self.input_fc1 = nn.Linear(1, self.nhid, bias=True)
        self.input_fc2 = nn.Linear(self.nhid, self.nhid, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        self.spa_GCN = GCN(self.nhid, self.nhid, dropout, alpha)
        self.cxt_GCN = GCN(self.nhid, self.nhid, dropout, alpha)
        
        self.pred_fc = nn.Linear(self.nhid, 1, bias=True)
        
        self.loss_r = nn.L1Loss()
        
        # Hierarchical graph neural network
        self.region_gcn = GCN(self.nhid+7, self.nhid, dropout, alpha)
        self.rz_gcn = GCN_2D(23, 12*11, dropout, alpha)
        self.zone_gcn = GCN(self.nhid, self.nhid, dropout, alpha)
        self.gate_z = nn.Linear(23+5, self.nhid, bias=True)
        self.zc_gcn = GCN(self.nhid, 11, dropout, alpha)
        self.city_gcn = GCN(self.nhid, self.nhid, dropout, alpha)
        self.gate_c = nn.Linear(23+5, self.nhid, bias=True)
        self.gru_layer = nn.Linear(self.nhid*3, self.nhid, bias=True)
        
        # Self-supervision
        self.n_GCN = GCN(self.nhid, self.nhid, dropout, alpha)
        self.neighPred_loss = N_Loss()
        self.cxtPred_layer = nn.Linear(self.nhid, 5, bias=True)
        
        # GRU cell
        self.GRU_app = nn.GRUCell(self.nhid, self.nhid, bias=True)
        nn.init.xavier_uniform_(self.GRU_app.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_app.weight_hh, gain=math.sqrt(2.0))
        
        self.GRU_hier = nn.GRUCell(self.nhid, self.nhid, bias=True)
        nn.init.xavier_uniform_(self.GRU_hier.weight_ih, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU_hier.weight_hh, gain=math.sqrt(2.0))
        
        # Prediction layer
        self.pred_layer = nn.Linear(self.nhid+24*5, 24, bias=True)
        
        # Parameter initialization
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight,gain=math.sqrt(2.0))
    
    def forward(self, X_batch, weather_forecast, spatial_attr, supports):
        
        B, N_total, T, F = X_batch.size()
        
        weather_forecast = weather_forecast.reshape((B, N_total, -1)) # (B, N, T*F)
        
        # to init GRU hidden state
        x_tem = torch.zeros(B, N_total, self.nhid).to(self.device)
        h_t = torch.zeros(B, N_total, self.nhid).to(self.device)
        
        reg_loss = 0
        ssl_loss = 0
        for i in range(T):
            # AQI distribution approximation
            x = self.input_fc1(X_batch[:, self.idx_label+self.idx_unlabel, i, 0:1]) # (B, N, nhid)
            x = self.leakyrelu(x)
            x = self.input_fc2(x) # (B, N, nhid)
            
            # Spatial view
            x_spa = self.spa_GCN(x[:, self.idx_label, :], supports[0][:, self.idx_label]) # (B, N, nhid)
            
            # Temporal view
            x_tem = self.GRU_app(x_spa.view(-1, self.nhid), x_tem.view(-1, self.nhid)) # (B*N, nhid)
            x_tem = x_tem.view(B, N_total, -1) # (B*N, nhid)
            
            # Contextual view
            x_cxt = self.cxt_GCN(x[:, self.idx_label, :], supports[1][:, self.idx_label]) # (B, N, nhid)
            
            # Fusion
            x_spa = x_spa.unsqueeze(-1) # (B, N, nhid, 1)
            x_tem = x_tem.unsqueeze(-1) # (B, N, nhid, 1)
            x_cxt = x_cxt.unsqueeze(-1) # (B, N, nhid, 1)
            x_out = torch.concat([x_spa, x_tem, x_cxt], dim=-1) # (B, N, nhid, 3)
            x_out = torch.mean(x_out, dim=-1) # (B, N, nhid)
            
            y_pred = self.pred_fc(x_out)

            # regression task
            reg_loss += self.loss_r(y_pred[:, self.idx_label].view(-1, 1), \
                                    X_batch[:, self.idx_label, i, 0].view(-1, 1))
            
            # classification task
            # y_pred_discrete = self.classified_fc(x_out)
            # class_loss += self.loss_c(y_pred_discrete[:, self.idx_label].view(-1, 1), \
            #                           X_batch[:, self.idx_label, i, 1].view(-1, 1))
            
            # Spatio-temporal self-supervision
            x_n = self.n_GCN(x_out, supports[2])
            if i==0:
                self.neighPred_loss.get_positive_node_pair(supports[2], 1000)
                self.neighPred_loss.get_negative_node_pair(supports[2], 20000)
            n_loss = self.neighPred_loss.get_loss(x_n, self.nhid)
            
            flow_pred = self.cxtPred_layer(x_out)
            c_loss = self.loss_r(flow_pred, X_batch[:, :, i, 8:])
            ssl_loss += (n_loss+c_loss)
            
            # Modeling region dependency
            x_rt = torch.concat([x_out, X_batch[:, :, i, 1:8]], dim=-1)
            x_rt = self.region_gcn(x_rt, supports[2])
            
            # Modeling functional zone dependency
            s_rz = self.rz_gcn(spatial_attr, supports[2]) # (N, zone_num)
            adj_rz = torch.softmax(torch.mul(s_rz, supports[4]), dim=-1) # (N, zone_num)
            x_z = torch.einsum('vw,bwf->bvf', (adj_rz.T.float(), x_out)) # (zone_num, N) (B, N, nhid)
            # x_z = apply_bn(x_z)
            adj_z = torch.mm(torch.mm(adj_rz.T.float(), supports[2].float()), adj_rz.float()) # (zone_num, zone_num)
            x_z_prime = self.zone_gcn(x_z, adj_z)
            
            spatial_gate = spatial_attr.unsqueeze(0).repeat(B, 1, 1)
            g_z = torch.sigmoid(self.gate_z(torch.concat([spatial_gate, X_batch[:, :, i, 1:6]], dim=-1).float()))
            x_rz = torch.einsum('vw,bwf->bvf', (adj_rz.float(), x_z_prime.float())) # (N, zone_num) (B, zone_num, nhid)
            x_rz = torch.mul(x_rz, g_z) # gating mechanism
            
            # Modeling city dependency
            s_zc = self.zc_gcn(x_z, adj_z) # (B, zone_num, city_num)
            adj_zc = torch.softmax(torch.mul(s_zc, supports[5]), dim=-1) # (B, zone_num, city_num)
            x_c = torch.einsum('bvw,bwf->bvf', (adj_zc.permute(0, 2, 1).float(), x_z)) # (B, city_num, nhid)
            x_c = self.city_gcn(x_c, supports[3])
            
            g_c = torch.sigmoid(self.gate_c(torch.concat([spatial_gate, X_batch[:, :, i, 1:6]], dim=-1).float()))
            x_rc = torch.einsum('bvw,bwf->bvf', (adj_zc.float(), x_c.float())) # (B, zone_num, nhid)
            x_rc = torch.einsum('vw,bwf->bvf', (adj_rz.float(), x_rc)) # (B, N, nhid)
            x_rc = torch.mul(x_rc, g_c) # gating mechanism

            # Modeling temporal dependency
            x_t = torch.concat([x_rt, x_rz, x_rc], dim=-1)
            x_t = self.gru_layer(x_t)
            h_t = self.GRU_hier(x_t.view(-1, self.nhid), h_t.view(-1, self.nhid)) # (B*N, nhid)
            h_t = h_t.view(B, N_total, -1) # (B, N, nhid)
        
        # Prediction
        h_t = torch.concat([h_t, weather_forecast], dim=-1)
        out = self.pred_layer(h_t)
        
        reg_loss = reg_loss/T
        ssl_loss = ssl_loss/T
        
        return out, reg_loss, ssl_loss
