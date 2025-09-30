import argparse
import time
import torch
import torch.utils.data
from mdanalysis.dataset import MDAnalysisDataset, collate_mda

from models.model import AGLSTAN, STAG, STGCN
from models.model_x import *
import os
from torch import nn, optim
import json
import pickle

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.stegmn.STEGMN import STEGMN
from setproctitle import setproctitle

parser = argparse.ArgumentParser(description='STEGMN')
parser.add_argument('--batch_size', type=int, default=100,
                    help='input batch size for training')
parser.add_argument('--device', type=str, default="cuda:0",
                    help='device for training')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='logs/mdanalysis_logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--nf', type=int, default=16, metavar='N',
                    help='hidden dim')

parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='weight decay')
parser.add_argument('--data_dir', type=str, default='mdanalysis',
                    help='Data directory.')


parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')   
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--exp_name', type=str, default='exp_1000', metavar='N', help='experiment_name')
parser.add_argument('--num_past', type=int, default=10,
                    help='Number of length of whole past time series.')
parser.add_argument('--time_point', type=int, default=5,
                    help='Time point of past time series (egnn):1,5,10.')
parser.add_argument('--delta_frame', type=int, default=5,
                    help='Number of frames delta.')
parser.add_argument('--model', type=str, default='egnn', metavar='N',
                    help='available models: baseline, egnn,stagmd')
parser.add_argument('--lr', type=float, default=5e-5, metavar='N',
                    help='learning rate')
parser.add_argument('--fft', type=eval, default=False,
                    help='Use FFT ')
parser.add_argument('--eat', type=eval, default=True,
                    help='Use EAT')   
parser.add_argument("--load_cached", action="store_true", help="Load cached dataset.")
parser.add_argument('--with_mask', action='store_true', default=False,
                    help='mask the future frame if use eat') 
parser.add_argument('--save_m', type=eval, default=True, help='whether to save model')  
parser.add_argument('--tempo', type=eval, default=True, help='Use temporal pooling') 




args = parser.parse_args()

setproctitle(args.exp_name)

args.cuda = not args.no_cuda and torch.cuda.is_available()


def calculate_r2_score(y_true, y_pred):
    """计算 R2 分数"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def calculate_mae(y_true, y_pred):
    """计算平均绝对误差"""
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()


def safe_save_model(model, path):
    """安全保存模型，避免 JIT 脚本函数的 pickle 问题"""
    try:
        # 尝试保存整个模型对象
        torch.save(model, path)
    except (pickle.PickleError, RuntimeError) as e:
        if "ScriptFunction cannot be pickled" in str(e) or "cannot pickle" in str(e):
            print(f"警告：模型包含无法序列化的组件，改为保存 state_dict: {e}")
            # 保存 state_dict 而不是整个模型
            torch.save(model.state_dict(), path)
        else:
            # 其他错误直接抛出
            raise e

torch.cuda.set_device(0)
# device = torch.device("cuda" if args.cuda else "cpu")
# device = torch.device("cpu")
device = torch.device(args.device)
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    # exp_path = args.outf + f"/exp_{args.model}"
    exp_path = args.outf + "/" + args.exp_name
    os.makedirs(exp_path)
except OSError:
    pass



def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  
    dataset_train = MDAnalysisDataset('adk', partition='train', tmp_dir=args.data_dir,
                                      delta_frame=args.delta_frame, load_cached=args.load_cached, num_past=args.num_past)
    sampler = None
    shuffle = True
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=shuffle, sampler=sampler, drop_last=True,
                                               num_workers=0, collate_fn=collate_mda)

    
    dataset_val = MDAnalysisDataset('adk', partition='valid', tmp_dir=args.data_dir,
                                    delta_frame=args.delta_frame, load_cached=args.load_cached, num_past=args.num_past)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=shuffle,
                                             drop_last=True, num_workers=0, collate_fn=collate_mda)

    # Val and test do not need sampler.

    dataset_test = MDAnalysisDataset('adk', partition='test', tmp_dir=args.data_dir,
                                     delta_frame=args.delta_frame, load_cached=args.load_cached,
                                     test_rot=False, test_trans=False, num_past=args.num_past)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                              shuffle=shuffle,  drop_last=True,
                                              num_workers=0, collate_fn=collate_mda)

    # 输出数据集长度信息
    print("=" * 60)
    print("Dataset Size Information")
    print("=" * 60)
    print(f"Protein: adk")
    print(f"Train set size: {len(dataset_train)}")
    print(f"Validation set size: {len(dataset_val)}")
    print(f"Test set size: {len(dataset_test)}")
    print(f"Total samples: {len(dataset_train) + len(dataset_val) + len(dataset_test)}")
    print(f"Train batches: {len(loader_train)}")
    print(f"Val batches: {len(loader_val)}")
    print(f"Test batches: {len(loader_test)}")
    
    # 输出节点数和边数信息
    print(f"\nNode and Edge Information:")
    # print(f"Number of nodes (simplified): {n_nodes}")
    print(f"Number of edges: {dataset_train.edge_attr.shape[0]}")
    print(f"Edge attribute dimension: {dataset_train.edge_attr.shape[1]}")
    



    #Adj=dataset_train.A.to(device)
    # start_time = time.time()
    # print("----Train (baseline)----:")
    # res = {'loss1': 0, 'loss5': 0, 'loss10': 0,  'counter': 0}
    # for batch_idx, data in enumerate(loader_train):
    #     loc,_, _ , _ , loc_end = data

    #     res['loss1'] += loss_mse(loc[0,:,:,:],loc_end).item()*args.batch_size
    #     res['loss5'] += loss_mse(loc[4,:,:,:],loc_end).item()*args.batch_size
    #     res['loss10'] += loss_mse(loc[9,:,:,:],loc_end).item()*args.batch_size
    #     res['counter'] += args.batch_size
    
    # print("Point 1: %.6f"%(res['loss1'] / res['counter']))
    # print("Point 5: %.6f"%(res['loss5'] / res['counter']))
    # print("Point 10: %.6f"%(res['loss10'] / res['counter']))

    #Point 1: 2.707518  Point 5: 2.408208  Point 10: 1.704617

    # start_time = time.time()
    # print("----Test (baseline)----:")
    # res = {'loss1': 0, 'loss5': 0, 'loss10': 0,  'counter': 0}
    # for batch_idx, data in enumerate(loader_test):
    #     loc, vel, edge_attr, charges, loc_end, edge_attr_fft, Fs_fft= data

    #     res['loss1'] += loss_mse(loc[0,:,:,:],loc_end).item()*args.batch_size
    #     res['loss5'] += loss_mse(loc[4,:,:,:],loc_end).item()*args.batch_size
    #     res['loss10'] += loss_mse(loc[9,:,:,:],loc_end).item()*args.batch_size
    #     res['counter'] += args.batch_size
    
    # print("Point 1: %.6f"%(res['loss1'] / res['counter']))
    # print("Point 5: %.6f"%(res['loss5'] / res['counter']))
    # print("Point 10: %.6f"%(res['loss10'] / res['counter']))

    # end_time = time.time()
    # print(end_time-start_time)
    # assert False

    #Point 1: 3.259700  Point 5: 3.301948  Point 10: 2.021766'''

    in_edge_nf = dataset_train.edge_attr.shape[-1]
    nodes_att_dim = 0

    n_nodes = dataset_train.charges.shape[0]

    print("in_edge_nf",in_edge_nf)


    if args.model=='egnn':
        model = EGNN_X( num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf, hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='stegmn':
        model = STEGMN(num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf, hidden_nf=args.nf, fft=args.fft, eat=args.eat, device=device, n_layers=args.n_layers, n_nodes=n_nodes, with_mask=args.with_mask, tempo=args.tempo, nodes_att_dim=nodes_att_dim)
    elif args.model=='gmn':
        model = GMN( num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf,hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='baseline': # past 1 --> future 1
        model = EGNN_X( num_past=1, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf,hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='stag_neq': # None-Equivariant STAG 
        out_dim = 4*3
        input_dim = 3*4+4
        model = STAG(num_nodes = n_nodes, num_features = input_dim, num_timesteps_input=args.num_past,num_timesteps_output=1, out_dim=out_dim).to(device)
    elif args.model=='gnn':
        input_dim = 3*4+4
        model = GNN_X(num_past=args.num_past, num_future=1, input_dim=input_dim, in_edge_nf=in_edge_nf, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model=='stgcn':
        num_features = 3*4+4
        out_dim = 3*4
        model = STGCN(num_nodes=n_nodes, num_features=num_features, num_timesteps_input=args.num_past,num_timesteps_output=1, out_dim=out_dim, device=device)
    elif args.model == 'se3_transformer' or args.model == 'tfn':
        from se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        model = SE3_Transformer(num_past=args.num_past, num_future=1, n_particles=n_nodes, n_dimesnion=3, nf=int(args.nf/args.degree), n_layers=args.n_layers, model=args.model, num_degrees=args.degree, div=1, device=device)
    elif args.model=='aglstan':
        num_features = 3*4+4
        out_dim = 3*4
        model = AGLSTAN(num_nodes=n_nodes, batch_size=args.batch_size, input_dim=num_features, output_dim=out_dim, window=args.num_past, num_layers=args.n_layers, filter_size=32, embed_dim=args.nf, cheb_k=3)
        model.to(device)
    else:
        raise Exception("Wrong model specified")




    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(0.2 * args.epochs), num_training_steps=args.epochs)


    results = {'epochs': [], 'test loss': [], 'val loss': [], 'train loss': [], 
               'test r2': [], 'val r2': [], 'train r2': [],
               'test mae': [], 'val mae': [], 'train mae': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    best_train_r2 = -1e8
    best_val_r2 = -1e8
    best_test_r2 = -1e8
    best_train_mae = 1e8
    best_val_mae = 1e8
    best_test_mae = 1e8
    
    # 输出详细的数据集信息
    print("=" * 60)
    print("Detailed Dataset Information")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden dimension: {args.nf}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Number of past frames: {args.num_past}")
    print(f"Delta frame: {args.delta_frame}")
    print(f"Test interval: {args.test_interval}")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of edges: {dataset_train.edge_attr.shape[0]}")
    print(f"Edge attribute dimension: {dataset_train.edge_attr.shape[1]}")
    print(f"Input edge features: {in_edge_nf}")
    print("=" * 60)

    #test_loss = train(model, optimizer, 5, loader_test, backprop=False)
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_r2, train_mae = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        results['train r2'].append(train_r2)
        results['train mae'].append(train_mae)
        if epoch % args.test_interval == 0:
            val_loss, val_r2, val_mae = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, test_r2, test_mae = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            results['val r2'].append(val_r2)
            results['test r2'].append(test_r2)
            results['val mae'].append(val_mae)
            results['test mae'].append(test_mae)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                best_train_r2 = train_r2
                best_val_r2 = val_r2
                best_test_r2 = test_r2
                best_train_mae = train_mae
                best_val_mae = val_mae
                best_test_mae = test_mae
                if args.save_m:
                    safe_save_model(model, f'{exp_path}/saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))
            print("*** Best Val R2: %.5f \t Best Test R2: %.5f \t Best Train R2: %.5f"
                  % (best_val_r2, best_test_r2, best_train_r2))
            print("*** Best Val MAE: %.5f \t Best Test MAE: %.5f \t Best Train MAE: %.5f"
                  % (best_val_mae, best_test_mae, best_train_mae))

        # scheduler.step()
        json_object = json.dumps(results, indent=4)
        with open(f"{exp_path}/loss.json", "w") as outfile:
            outfile.write(json_object)

    end_time = time.time()
    print(f'************training time(s) of {args.model}: {(end_time-start_time) / args.epochs}')
    return best_train_loss, best_val_loss, best_test_loss, best_epoch, best_train_r2, best_val_r2, best_test_r2, best_train_mae, best_val_mae, best_test_mae


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0}
    
    # 用于计算 R2 和 MAE 的累积数据
    all_preds = []
    all_targets = []
    
    # 输出批次信息（仅在第一个epoch和第一个批次时）
    if epoch == 0 and len(loader) > 0:
        print(f"Processing {len(loader)} batches with batch_size={loader.batch_size}")
        print(f"Total samples in this loader: {len(loader.dataset)}")

    for batch_idx, data in tqdm(enumerate(loader)):
        batch_size, n_nodes = args.batch_size, data[0].shape[1]//args.batch_size
        data = [d.to(device) if torch.is_tensor(d) else d for d in data]

        loc, edge_attr, charges, loc_end, edge_attr_fft, Fs_fft, vec, cfg = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = MDAnalysisDataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}


        #print(loss_mse(torch.mean(loc,axis=0),loc_end))
        optimizer.zero_grad()


        if args.model=='baseline':
            loc_pred=model(charges, loc[args.time_point-1].unsqueeze(0), edges, edge_attr)
        elif args.model=='stag_neq':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            node = feature.permute(1,0,2).reshape(batch_size,n_nodes,feature.shape[0],feature.shape[2])
            Adj = loader.dataset.A.to(device)

            loc_pred = loc[-1] + model(Adj, node).reshape(loc.shape[1],4,3)
        elif args.model=='gmn':
            loc_pred=model(charges, loc, edges, edge_attr)
        elif args.model=='stegmn':
            loc_pred=model(charges, loc, edges, edge_attr, vec, cfg)
        elif args.model=='egnn':
            loc_pred=model(charges, loc, edges, edge_attr)
        elif args.model=='gnn':
            nodes = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model=='stgcn':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)

            node = feature.permute(1,0,2).reshape(batch_size,n_nodes,feature.shape[0],feature.shape[2])
            Adj = loader.dataset.A.to(device)

            loc_pred = loc[-1] + model(Adj, node).reshape(loc.shape[1],4,3)
            #loc_pred = model(Adj, node).reshape(loc.shape[1],4,3)
        elif args.model == 'aglstan':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            node = feature.permute(1,0,2).reshape(batch_size,feature.shape[0], n_nodes, feature.shape[2])

            loc_pred = model(node)
            # loc_pred = loc[-1] + loc_pred.reshape(loc.shape[1], 4, 3)
            loc_pred = loc_pred.reshape(loc.shape[1], 4, 3)
        else:
            raise Exception("Wrong model")


        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        
        # 收集预测值和目标值用于计算 R2 和 MAE
        all_preds.append(loc_pred.detach())
        all_targets.append(loc_end.detach())

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    
    # 计算 R2 和 MAE 指标
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 将数据展平以计算指标
        preds_flat = all_preds.reshape(-1)
        targets_flat = all_targets.reshape(-1)
        
        r2 = calculate_r2_score(targets_flat, preds_flat)
        mae = calculate_mae(targets_flat, preds_flat)
    else:
        r2 = 0.0
        mae = 0.0
    
    print('%s epoch %d avg loss: %.7f R2: %.7f MAE: %.7f' % 
          (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], r2, mae))

    return res['loss'] / res['counter'], r2, mae


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch, best_train_r2, best_val_r2, best_test_r2, best_train_mae, best_val_mae, best_test_mae = main()
    print("best_train_loss = %.8f" % best_train_loss)
    print("best_val_loss = %.8f" % best_val_loss)
    print("best_test_loss = %.8f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    print("best_train_r2 = %.8f" % best_train_r2)
    print("best_val_r2 = %.8f" % best_val_r2)
    print("best_test_r2 = %.8f" % best_test_r2)
    print("best_train_mae = %.8f" % best_train_mae)
    print("best_val_mae = %.8f" % best_val_mae)
    print("best_test_mae = %.8f" % best_test_mae)
    with open(f"{exp_path}/loss.json") as f:
        loss=json.load(f)
    #plt.plot(loss['train loss'],label='Train')
    plt.plot(loss['epochs'],[np.mean(loss['train loss'][i*5:(i+1)*5]) for i in range(len(loss['train loss'])//args.test_interval )],label='Train')
    plt.plot(loss['epochs'],loss['test loss'],label='Test')
    plt.legend()
    plt.title("Loss")
    plt.savefig(f'{exp_path}/loss.png')



