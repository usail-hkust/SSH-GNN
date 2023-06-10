import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data', help='data path')
parser.add_argument('--adjdata', type=str, default='data/graph_adj', help='adj data path')
parser.add_argument('--nhid', type=int, default=64,help='')
parser.add_argument('--num_labeled_nodes', type=int, default=479, help='number of labeled nodes')
parser.add_argument('--num_nodes', type=int, default=2405, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--dropout', type=float,default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int,default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()

def main():
    # load data
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    spatial_attr = torch.tensor(dataloader['spatial_attr']).to(device)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    engine = trainer(scaler, args.num_labeled_nodes, args.num_nodes, args.nhid, args.dropout, \
                     args.learning_rate, args.weight_decay, device, spatial_attr, supports)

    print("start training...", flush=True)
    his_loss =[]
    test_time = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs+1):
        # training
        train_loss = []
        train_mae = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device) # (B, T, N, F)
            trainx = trainx.transpose(1, 2) # (B, N, T, F)
            trainy = torch.Tensor(y).to(device) # (B, T, N, F)
            trainy = trainy.transpose(1, 2) # (B, N, T, F)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mae[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        
        # validation
        valid_loss = []
        valid_mae = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 2)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 2)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_rmse, (t2 - t1)),flush=True)

    # testing
    test_loss = []
    test_mae = []
    test_rmse = []

    p1 = time.time()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 2)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 2)
        metrics = engine.eval(testx, testy)
        test_loss.append(metrics[0])
        test_mae.append(metrics[1])
        test_rmse.append(metrics[2])
    p2 = time.time()
    log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
    print(log.format(i,(p2-p1)))
    test_time.append(p2-p1)

    mtest_loss = np.mean(test_loss)
    mtest_mae = np.mean(test_mae)
    mtest_rmse = np.mean(test_rmse)
    his_loss.append(mtest_loss)

    log = 'Epoch: {:03d}, Test Loss: {:.4f}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test Time: {:.4f}/epoch'
    print(log.format(i, mtest_loss, mtest_mae, mtest_rmse, (p2 - p1)),flush=True)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
