import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_adj(filename):
    """Load adjacency matrices."""
    # normalized proximity matrix based on geographical distance
    distance_adj = np.load(filename+'/distance_adj.npy') # BTH (2405, 479)
    
    # normalized proximity matrix based on semantic similarity
    context_adj = np.load(filename+'/context_adj.npy') # BTH (2405, 479)
    
    # region adjacency matrix
    region_adj = np.load(filename+'/region_adj.npy') # BTH (2405, 2405)
    region_adj = calculate_normalized_laplacian(region_adj).astype(np.float32).todense()
    
    # city adjacency matrix
    city_adj = np.load(filename+'/city_adj.npy') # BTH (11, 11)
    city_adj = calculate_normalized_laplacian(city_adj).astype(np.float32).todense()
    
    # region-to-zone indication matrix
    rz_assign_adj = np.load(filename+'/rz_assign_adj.npy') # BTH (2405, 12*11)
    
    # zone-to-city indication matrix
    zc_assign_adj = np.load(filename+'/zc_assign_adj.npy') # BTH (12*11, 11)
    
    adj = [distance_adj, context_adj, region_adj, city_adj, rz_assign_adj, zc_assign_adj]
    
    return adj

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    """Load datasets."""
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        # (batch, historical time steps, nodes, features) BTH (batch, 12, 2405, 13)
        # feature dimension: 13, 0: air quality, 1-5: weather condition, 6-7: time of day and day of week, 8-12: traffic flow
        data['x_' + category] = cat_data['x']
        # (batch, forecast horizon, nodes, features) BTH (batch, 24, 2405, 6)
        # feature dimension: 6, 0: air quality, 1-5: weather forecast
        data['y_' + category] = cat_data['y']
        print('*'*10, category, data['x_' + category].shape, data['y_' + category].shape, '*'*10)
    
    # Data format
    for i in range(6):
        scaler = StandardScaler(mean=data['x_train'][..., i].mean(), std=data['x_train'][..., i].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i])
            
    for i in range(1, 6):
        scaler = StandardScaler(mean=data['y_train'][..., i].mean(), std=data['y_train'][..., i].std())
        for category in ['train', 'val', 'test']:
            data['y_' + category][..., i] = scaler.transform(data['y_' + category][..., i])
    
    # save the mean and std of air quality
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    
    # Load spatial attributes, i.e., POI features and road network features
    spatial_attr = np.load(dataset_dir+'/spatial_attr.npy') # (Nodes, Features)
    print('*'*10, 'spatial_attr', spatial_attr.shape, '*'*10)
    
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    data['spatial_attr'] = spatial_attr
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse
