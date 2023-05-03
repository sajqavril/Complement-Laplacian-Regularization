import os
from pickletools import optimize
from random import sample
import numpy as np
import pandas
import torch_geometric
import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
import pandas as pd

from model import MLP, GCN_Net, SAGE_Net, GAT_Net, Cheb_Net
from utils import get_data, set_best_train_args, sample_complement, lap_loss, normalize_edge, square_p_loss, mad_reg_loss
from torch.utils.tensorboard import SummaryWriter


def one_run(args, seed, run, bar, writer):
    
    torch.manual_seed(seed)
    torch_geometric.seed_everything(seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(seed)
    data = get_data(args) # all the topological data are sparsed
    if args.net == 'mlp':
        model = MLP(ninput=data.x.shape[1], nclass=data.y.max()+1, args=args)
    elif args.net == 'gcn':
        model = GCN_Net(ninput=data.x.shape[1], nclass=data.y.max().item()+1, args=args, pred=True)
    elif args.net == 'gat':
        model = GAT_Net(ninput=data.x.shape[1], nclass=data.y.max().item()+1, args=args, pred=True)
    elif args.net == 'sage':
        model = SAGE_Net(ninput=data.x.shape[1], nclass=data.y.max().item()+1, args=args, pred=True)
    elif args.net[:4] == 'cheb':
        model = Cheb_Net(ninput=data.x.shape[1], nclass=data.y.max().item()+1, nhid=args.nhid, K=int(args.net[4]))

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
        data = data.cuda()
        model = model.cuda() 

    else:
        device = torch.device('cpu')
    degree = data.edge_index.shape[1] / data.x.shape[0]

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}])

    if not args.no_earlystop:
        best_epoch = 0
        best_val_acc = 0.
        bad_epochs = 0
        best_test_acc = 0.
        best_val_loss = torch.inf
    
    for ite in range(args.epochs):
        bar.set_description('Run:{:2d}, iter:{:4d}'.format(run, ite))
        
        model.train()
        optimizer.zero_grad()
        pred = model(data)
        output = F.log_softmax(pred, dim=1)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        IMPORTANT TO UNDERSTANDING THE VARIOUS TYPES OF REGULARIZATION METHODS
        here we could append various types of regularization methods:
        MADReg: mad_reg_loss
        P_reg: square_p_loss
        NL_reg: only give positive values to \alpha and set \beta to zero
        CLAR_reg: adjusting \alpha and \beta together
        '''

        # soft_loss = square_p_loss(pred, data.edge_index).mean().clamp(0., 1.)
        # soft_loss = mad_reg_loss(pred, data.edge_index).clamp(0., 1.)
        # soft_loss = soft_loss * (-0.5) # to be adjusted as the original paper 
        # soft_loss.backward(retain_graph=True)

        if args.neg > 0.:    
            neg_edges, neg_norm = sample_complement(data.edge_index, data.x.shape[0], s=args.S)
            neg_loss = lap_loss(pred, neg_edges, neg_norm).clamp(0., 1.) 
            (neg_loss * args.neg).backward(retain_graph=True)
            del neg_loss, neg_norm
        if args.pos > 0.:
            pos_norm = normalize_edge(edge_index=data.edge_index, num_nodes=data.x.shape[0])
            pos_loss = lap_loss(pred, data.edge_index, pos_norm).clamp(0., 1.)
            (pos_loss * args.pos).backward(retain_graph=True)
            del pos_loss, pos_norm

        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        '''

        loss.backward()
        optimizer.step()

        model.eval()
        eval_pred = model(data)
        eval_output = F.log_softmax(eval_pred, dim=1)
        
        val_loss = F.nll_loss(eval_output[data.val_mask], data.y[data.val_mask])
        val_acc = (eval_output.argmax(dim=1) == data.y)[data.val_mask].sum() / data.val_mask.sum()
        test_acc = (eval_output.argmax(dim=1) == data.y)[data.test_mask].sum() / data.test_mask.sum()

        bar.set_postfix(train_loss='{:.4f}'.format(loss.item()), 
                        val_loss='{:.4f}'.format(val_loss.item()),
                        val_acc='{:.4f}'.format(val_acc.item()))
        # print('Epoch %d: train loss: %.4f, val loss: %.4f, val acc: %.4f, test acc %.4f'%(epoch, loss, val_loss, val_acc, test_acc))

        writer.add_scalar('{:s}/Loss-Train'.format(args.dataset, args.net), loss.item(), ite)
        writer.add_scalar('{:s}/Loss-Val'.format(args.dataset, args.net), val_loss.item(), ite)
        writer.add_scalar('{:s}/Accuracy-Val'.format(args.dataset, args.net), val_acc.item(), ite)
        writer.add_scalar('{:s}/Accuracy-Test'.format(args.dataset, args.net), test_acc.item(), ite)


        if not args.no_earlystop:
            if val_loss < best_val_loss:
                best_epoch = ite
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc = test_acc
                bad_epochs = 0
            else:
                bad_epochs = bad_epochs + 1

            if bad_epochs >= args.patience and best_epoch > 50: # warm_up == 50
                print('\nBest epoch %d: train loss: %.4f, val loss: %.4f, val acc: %.4f, test acc %.4f'%(best_epoch, loss, best_val_loss, best_val_acc, best_test_acc))
                # print(model)
                break

    if args.no_earlystop:
        best_test_acc = test_acc
        best_val_acc = val_acc
    
    return best_test_acc.item(), best_val_acc.item()

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (no use).')
    parser.add_argument('--nhid', type=int, default=32, help='hidden layer')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--dataset', type=str, default='pubmed', help='Data set.')
    parser.add_argument('--no_earlystop', action='store_true', default=False, help='Set to voyage the whole epochs.')
    parser.add_argument('--patience', type=int, default=100,
                        help='Heads of distribution attention.')
    parser.add_argument('--runs', type=int, default=3,
                        help='Runs to train.')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate for feature transformation.')
    parser.add_argument('--drop_prop', type=float, default=0.,
                        help='Dropout rate for propagation.')
    parser.add_argument("--net", type=str, default='mlp', choices=['bern', 'beta', 'gprgnn', 'cheb', 'lag', 'gcn', 'lag2', 'mlp', 'gcn', 'gat', 'sage', 'cheb2', 'cheb4'])

    # beta parameters
    parser.add_argument('--alpha', type=int, default=2, help='Alpha in (1, 2, ..., alpha).')

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--init', type=str, default='sgc', choices=['sgc', 'ppr', 'nppr', 'random'])

    parser.add_argument('--relu', action='store_true', default=False, help='Use the RELU on thetas')
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'leaky', 'tanh', 'sigmoid', 'none'])

    # training parameters
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for linear')
    parser.add_argument('--prop_lr', type=float, default=0.01, help='Weight decay for linear')
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam', 'adamax', 'adagrad', 'asdg', 'delta'])

    # split parameters
    parser.add_argument('--split', type=str, default='random', choices=['random', 'set', 'grand'])
    parser.add_argument('--train_proportion', type=float, default=0.6, help='Train proportion')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='Valid proportion')
    parser.add_argument('--idx', type=int, default=0, help='For multiple graphs, e.g. ppi has 20 graphs')

    # best training params from bernnet
    parser.add_argument('--no_best_hyper_params', action='store_true', default=False, help='Use the best hyper parameters from bernnet')
    parser.add_argument('--sample', type=str, default='node', choices=['node', 'edge'])

    # clar
    parser.add_argument('--S', type=int, default=4)
    parser.add_argument('--pos', type=float, default=0.01)
    parser.add_argument('--neg', type=float, default=0.01)

    

    args = parser.parse_args()
    if args.dataset.lower() in ['cs', 'physics']:
        args.split = 'grand'
    elif args.dataset.lower() in ['computers', 'photo', 'chameleon', 'squirrel', 'actor', 'texas', 'cornell']:
        args.split = 'random'
    
    if not args.no_best_hyper_params:
        set_best_train_args(args)

    args.runs = 5
    args.nhid = 32

    print(args)

    torch.set_num_threads(1)

    writer = SummaryWriter(comment='_reg_pos_{:.4f}_neg_{:.4f}_S_{:d}_model_{:s}'.format(args.pos, args.neg, args.S, args.net))

    seeds=[0,1,2,3,4,5,6,7,8,9]

    pbar = tqdm.tqdm(range(args.runs))

    perm = torch.randperm(len(seeds))
    rand_seeds = torch.LongTensor(seeds)[perm]

    test_accs = []
    val_accs = []

    for idx in pbar:
        test_acc, val_acc = one_run(args, seed=rand_seeds[idx], run=idx, bar=pbar, writer=writer)
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print('Average Test acc for {:s}: {:.4f}, Val acc: {:.4f}'.format(args.dataset, torch.Tensor(test_accs).mean().item(), torch.Tensor(val_accs).mean().item()))
    res = {
        'pos': [args.pos],
        'neg': [args.neg],
        'S': [args.S],
        'test_acc': [torch.Tensor(test_accs).mean().item()]
    }
    df = pd.DataFrame(res)
    df.to_csv('{:s}_dataset_{:s}.csv'.format(args.net, args.dataset), mode='a+', header=False)
    
    writer.close()

if __name__ == '__main__':
    main()



    
    
    

