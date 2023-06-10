import torch.optim as optim
from model import *
import util

class trainer():
    def __init__(self, scaler, num_labeled_nodes, num_nodes, nhid, dropout, lrate, wdecay, device, spatial_attr, supports):
        label_num = round(num_labeled_nodes*0.7)
        self.idx_label = [int(i) for i in range(0, label_num)] # labeled regions
        self.idx_unlabel = [int(i) for i in range(label_num, num_labeled_nodes)] # masked unlabeled regions
        
        self.model = SSL_GNN(device, self.idx_label, self.idx_unlabel, nhid, dropout)
        self.model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.spatial_attr = spatial_attr # (N, F)
        self.supports = supports
        
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        # input: (B, N, T, F)
        # real_val: (B, N, T, F)
        self.model.train()
        self.optimizer.zero_grad()
        weather_forecast = real_val[:, :, :, 1:] # weather forecast features
        output, reg_loss, ssl_loss = self.model(input, weather_forecast, self.spatial_attr, self.supports) # output: (B, N, T)
        real = real_val[:, self.idx_label, :, 0] # (B, N, T)
        predict = self.scaler.inverse_transform(output[:, self.idx_label, :]) # (B, N, T)

        loss = self.loss(predict, real, 0.0)+0.5*reg_loss+0.1*ssl_loss
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.loss(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mae, rmse

    def eval(self, input, real_val):
        self.model.eval()
        weather_forecast = real_val[:, :, :, 1:]
        output, reg_loss, ssl_loss = self.model(input, weather_forecast, self.spatial_attr, self.supports)
        real = real_val[:, self.idx_unlabel, :, 0]
        predict = self.scaler.inverse_transform(output[:, self.idx_unlabel, :])
        
        loss = self.loss(predict, real, 0.0)+0.5*reg_loss+0.1*ssl_loss
        mae = self.loss(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        
        return loss.item(), mae, rmse
