import os
import torch
import numpy as np
from torch import device, sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import ChebConv
from torch_geometric.utils.loop import add_self_loops
import torch.nn.functional as F
import torch_sparse
import torch_geometric.transforms as T
import pandas as pd
from torch_scatter import scatter_add

from torch_geometric.utils import get_laplacian, subgraph, negative_sampling
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor, Coauthor, Amazon, PPI, Reddit2, Yelp, AmazonProducts
from model_test import GPR_prop, Bern_prop, Lagendre2_prop, Lagendre_prop, Cheb_prop
from ogb.nodeproppred import PygNodePropPredDataset

def get_gamma(x):

    assert x > 0

    if x >= 2 and x <= 3:
        base =  2. / (4. - x)

    elif x < 2:
        h = int(x-2)
        base = 2. / (4. - (x - h))
        for k in range(1, h + 1):
            base = base * (x - k)

    else: # x > 3
        h = int(3 - x)
        base = 2. / (4. - (x + h))
        for k in range(0, h):
            base = base / (x - k)

    return base


def successive_spspmm(sparse_Xs=[]):
    '''
    sparse_Xs are List of torch.sparse_coo_tensor with shapes:
    sparse_A: [A, D1]
    sparse_B: [D1, D2]
    sparse_C: [D2, C]
    ...
    reture torch.sparse_coo_tensor
    '''
    def simple_spspmm(sparse_A, sparse_B):
        assert sparse_A.size(1) == sparse_B.size(0)
        prod_indices, prod_values = torch_sparse.spspmm(
            sparse_A.coalesce().indices(), sparse_A.coalesce().values(),
            sparse_B.coalesce().indices(), sparse_B.coalesce().values(),
            sparse_A.size(0), sparse_A.size(1), sparse_B.size(1)
        )
        return prod_indices, prod_values, torch.Size([sparse_A.size(1), sparse_B.size(0)])

    product = sparse_Xs[0]

    for idx in range(1, len(sparse_Xs)):
        prod_indices, prod_values, prod_size = simple_spspmm(product, sparse_Xs[idx])
        product = torch.sparse_coo_tensor(
        indices=prod_indices,
        values=prod_values,
        size=prod_size,
        )

    return product

def get_data(args):

    name = args.dataset
    split = args.split
    train_proportion = args.train_proportion
    val_proportion = args.val_proportion
    idx = args.idx

    if name in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products', 'ogbn-papers100M']:
        transforms = T.Compose([T.NormalizeFeatures(), T.ToUndirected(), T.AddSelfLoops()])
        dataset = PygNodePropPredDataset(name=name, transform=transforms, root='/mnt/data_16TB/sjq20/data/')
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        data.y = data.y.flatten()
        # data.adj_t = data.adj_t.to_symmetric()
        data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0])

    elif name in ['yelp', 'ap', 'reddit']:
        transforms = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        if name == 'yelp':
            dataset = Yelp('/mnt/data_16TB/sjq20/data/Yelp', transform=transforms)
        elif name == 'reddit':
            dataset = Reddit2('/mnt/data_16TB/sjq20/data/Reddit2', transform=transforms)
        elif name == 'ap':
            dataset = AmazonProducts('/mnt/data_16TB/sjq20/data/AmazonProducts', transform=transforms)
        data = dataset[0]
        data.y = data.y.flatten()

    else:
        name = name.lower()
        path = os.path.join('.', 'data')
        if name == 'actor':
            path = os.path.join(path, name)
        transforms = T.Compose([T.NormalizeFeatures()])
        dataset_func = {
            'cora': Planetoid,
            'citeseer': Planetoid,
            'pubmed': Planetoid,
            'cornell': WebKB,
            'texas': WebKB,
            'wisconsin': WebKB,
            'actor': Actor,
            'computers': Amazon, 
            'photo': Amazon,
            'physics': Coauthor,
            'cs': Coauthor,
            'ppi': PPI       
        }
        data_npz = {
            'chameleon': '/home/sjq20/model/homophily-paper/data/chameleon/raw/chameleon.npz',
            'squirrel': '/home/sjq20/model/homophily-paper/data/squirrel/raw/squirrel.npz',
        }
        
        if name not in ["chameleon", "squirrel", "actor"]:
            dataset = dataset_func[name](path, name=name, transform=transforms)
            data = dataset[idx]
        elif name in ["actor"]:
            dataset = dataset_func[name](path, transform=transforms)
            data = dataset[idx]
        else:
            data = load_npz_data(data_npz[name])
        
        if (name in ['cora', 'pubmed', 'citeseer']) and (split == 'grand'):
            # data = Planetoid(path, name, transform=transforms, split='public')[0]
            pass

        else: # for now, one split is used
            num_class = data.y.max() + 1
            n = data.x.shape[0]
            val_lb = int(n * val_proportion)
            percls_trn = int(train_proportion * n / num_class)
            data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
        
    # if args.rand_edge:
    #     perm = torch.randperm(data.edge_index.shape[1])
    #     rand_edges = torch.stack([data.edge_index[0, perm], data.edge_index[1]], dim=0)
    #     data.edge_index = rand_edges
    # if args.iden_edge:
    #     data.edge_index = torch.ones(size=(2, data.x.shape[0])).type_as(data.edge_index)
    # # data.edge_index_sym = symmetric_normed_laplacian(data.x.shape[0], data.edge_index)
    # if args.drop_edge > 0.:
    #     perm = torch.randperm(data.edge_index.shape[1])
    #     perm = perm[:int(args.drop_edge * data.edge_index.shape[1])]
    #     data.edge_index = data.edge_index[:,perm]
    # if args.add_edge > 0.:
    #     idx = torch.randperm(data.x.shape[0])
    #     head = idx[:int(args.add_edge * data.x.shape[0])]
    #     tail = idx[-int(args.add_edge * data.x.shape[0]):]
    #     heads = torch.cat([data.edge_index[0], head], dim=0).type_as(data.edge_index)
    #     tails = torch.cat([data.edge_index[1], tail], dim=0).type_as(data.edge_index)
    #     data.edge_index = torch.stack([heads, tails], dim=0)
    # data.edge_index_sym = symmetric_normed_laplacian(data.x.shape[0], data.edge_index)

    # smooth_x = feature_smooth(data.x, data.edge_index, args.smooth, args.S, args.dataset)
    # base = smooth_x.norm(dim=1, p=2, keepdim=True)
    # base[base == 0.] = 1.
    # data.x = smooth_x.div(base)
    # data.base_x = data.x

    # inductive_idx = mask_to_index(data.train_mask + data.val_mask)
    # inductive_sub = subgraph(inductive_idx, data.edge_index, relabel_nodes=True)[0]
    # inductive_data = Data(x=data.x[inductive_idx], edge_index=inductive_sub, y=data.y[inductive_idx])
    # inductive_train_mask = data.train_mask[inductive_idx]
    # inductive_val_mask = data.val_mask[inductive_idx]

    # data.inductive_data = inductive_data
    # data.inductive_train_mask = inductive_train_mask
    # data.inductive_val_mask = inductive_val_mask

    return data

def sample_complement(edge_index, num_nodes, s=4, mode='edge', normalization=True):
    if mode == 'edge':
        neg_edges = []
        for i in range(s):
            neg_edge = negative_sampling(edge_index, num_nodes=num_nodes).to(edge_index.device)
            neg_edges.append(neg_edge)
            neg_edge = negative_sampling(edge_index[[1,0]], num_nodes=num_nodes).to(edge_index.device)
            neg_edges.append(neg_edge)
        neg_edges = torch.cat(neg_edges, dim=1)
        if normalization:
            neg_norm = normalize_edge(neg_edges, num_nodes)
        else:
            neg_norm = torch.ones_like(neg_edges[0], device=edge_index.device, dtype=torch.float)
        return neg_edges, neg_norm

def normalize_edge(edge_index, num_nodes):
    # normalize an adjacency
    row, col = edge_index[0], edge_index[1]
    edge_weight = torch.ones_like(row, device=row.device, dtype=torch.float)
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_weight

def lap_loss(x, indices, values):
    head_x = x[indices[0,:]]
    tail_x = x[indices[1,:]]
    loss = ((head_x - tail_x).norm(dim=1, p=2) * values).mean()
    return loss * loss * indices.shape[1]

def mask_to_index(mask):
    return mask.nonzero().flatten()

def get_batch(data, num, train_mask, sample='node'):

    assert data.x.shape[0] == train_mask.shape[0]

    if sample == 'node':
        batch_idx = torch.randperm(data.x.shape[0])[:num].cuda().sort().values
        batch_x = data.x[batch_idx]
        batch_y = data.y[batch_idx]
        batch_edge_index = subgraph(batch_idx, data.edge_index, relabel_nodes=True)[0]
        
    elif sample == 'edge':
        assert num <= data.edge_index.shape[1]
        edge_idx = torch.randperm(data.edge_index.shape[1])[:num].cuda()
        batch_edge_index = data.edge_index[:, edge_idx]
        batch_idx = mask_to_index(index_to_mask(batch_edge_index.flatten(), data.x.shape[0]))
        batch_x = data.x[batch_idx]
        batch_y = data.y[batch_idx]
        batch_edge_index = subgraph(batch_idx, data.edge_index, relabel_nodes=True)[0]
    
    batch_data = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y)
    batch_train_mask = train_mask[batch_idx]

    return batch_data, batch_train_mask

def split_data(y, split='random', train_proportion=0.6, val_proportion=0.2):
    '''
    data with x, y, edge_index
    connection is not assured
    random: randomly select (default:0.6/0.2/0.2, texas, cornell)
    set: 20/500/1000 
    '''

    n = y.shape[0]
 
    train_mask = torch.zeros(n).bool()
    val_mask = torch.zeros(n).bool()
    if split == 'random':
        train_per_class = int(train_proportion * n / (y.max() + 1))
        val_num = int(val_proportion * n)
        test_num = 0
    elif split == "set":
        train_per_class = 20
        val_num = 500
        test_num = 1000
    else: # split == 'grand'
        train_per_class = 20
        val_per_class = 30

    for c in range(y.max() + 1):
        class_idx = (y == c).nonzero().reshape(-1)
        perm = torch.randperm(class_idx.shape[0])
        class_idx = class_idx[perm]
        if class_idx.shape[0] > train_per_class:
            train_idx = class_idx[:train_per_class]
            train_mask[train_idx] = True
        else:
            train_mask[class_idx] = True
        if split == 'grand':
            val_idx = class_idx[train_per_class:(train_per_class + val_per_class)]
            val_mask[val_idx] = True

    if not split == 'grand':
        val_test_idx = (~train_mask).nonzero().squeeze()
        perm = torch.randperm(val_test_idx.shape[0])
        val_test_idx = val_test_idx[perm]
        val_idx = val_test_idx[:val_num]
        val_mask[val_idx] = True
        if split == 'set':
            test_mask = torch.zeros(n).bool()
            test_idx = val_test_idx[val_num:(val_num + test_num)]
            test_mask[test_idx] = True
        else: # random
            test_mask = ~(train_mask + val_mask)
    else: # grand
        test_mask = ~(train_mask + val_mask)
    
    return train_mask, val_mask, test_mask

def symmetric_normed_laplacian(n, edge_index):

    m = edge_index.shape[1]
    # symmetry the adjacency
    edge_index_sym = torch.sparse_coo_tensor(
            indices=torch.cat([edge_index, edge_index[[1,0]]], dim=1),
            values= torch.ones(size=[m * 2]),
            size=[n, n],
            device=edge_index.device).coalesce().indices()
        
    # indices, values = get_laplacian(edge_index_sym, normalization='sym', num_nodes=n)
    # indices, values = add_self_loops(indices, values, fill_value=-1., num_nodes=n)
    # laplacian = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])

    # #L=I-D^(-0.5)AD^(-0.5)
    # edge_index1, norm1 = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    # lap1 = torch.sparse_coo_tensor(indices=edge_index1, values=norm1, size=[n, n])
    # #2I-L
    # edge_index2, norm2 = add_self_loops(edge_index1, norm1, fill_value=2., num_nodes=n)
    # lap2 = torch.sparse_coo_tensor(indices=edge_index2, values=norm2, size=[n, n])

    # return laplacian, lap1, lap2, edge_index_sym
    return edge_index_sym

def load_npz_data(path):
    raw = dict(np.load(path, allow_pickle=True))
    data = Data(x=torch.Tensor(raw['features']), y=torch.LongTensor(raw['label']), edge_index=torch.LongTensor(raw['edges']).t())
    del raw

    return data

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def set_best_train_args(args):
    table = pd.read_csv('best_params.csv', delimiter=', ', engine='python')
    name = args.dataset.lower()
    
    ind = (table['dataset']==name).to_numpy().nonzero()[0]
    args.lr = float(table['lr'][ind])
    args.prop_lr = float(table['prop_lr'][ind])
    args.nhid = int(table['nhid'][ind])
    args.drop_prop = float(table['drop_prop'][ind])
    args.dropout = float(table['dropout'][ind])
    # args.K = int(table['K'][ind])
    args.weight_decay = float(table['weight_decay'][ind])
    return args

def similar_edge_index(x):
    inner_prod = torch.mm(x, x.T)
    
def feature_smooth(x, edge_index, method='none', S=1, dataset='cora'):

    if method == 'none':
        return x
    n = x.shape[0]
    if method in ['gcn', 'lap']:
        if method == 'gcn':
            smooth_index, smooth_norm = gcn_norm(edge_index, num_nodes=n)
        if method == 'lap':
            smooth_index, smooth_norm = get_laplacian(edge_index, normalization='sym', num_nodes=n)
        smooth_mat = torch.sparse_coo_tensor(indices=smooth_index, values=smooth_norm, size=(n, n))
        
        base_mat = smooth_mat.coalesce()
        for _ in range(S-1):
            smooth_mat = torch.sparse.mm(base_mat, smooth_mat).coalesce()
        
        return torch.sparse.mm(smooth_mat, x)
    
    prop_map = {
            'cheb': Cheb_prop,
            'gprgnn': GPR_prop,
            'bern': Bern_prop,
            'lag': Lagendre_prop,
            'lag2': Lagendre2_prop,
        }
    prop = prop_map[method](S, temp_grad=False)
    x = prop(x, edge_index)
    del prop

    # smooth_mat = torch.load('/home/sjq20/model/homophily-paper/output/F_{:s}_bern.pkl'.format(dataset)).detach().to(x.device)
    # x = torch.mm(smooth_mat, x)
    # del smooth_mat

    return x

def s_loss(FX, Y, delta=1, p=2, train_mask=None):
    '''
    FX \in  \mathtt{R}^{n \times c}
    Y \in \mathtt{R}^{n \times 1}
    '''
    n = Y.shape[0] 
    c = Y.max() + 1
    
    D = torch.cdist(FX, FX, p=2)
    # D = torch.mm(FX, FX.T)
    S = torch.softmax(D / (-2 * delta), dim=1)
    # P = torch.ones((n, c)).cuda() / c


    return S.log_softmax(dim=1)

def mask_onehot(Y, mask=None):
    n = Y.shape[0]
    c = Y.max() + 1
    Y = Y.to_sparse()
    onehot = torch.sparse_coo_tensor(
        indices=torch.stack([Y.indices()[0], Y.values()], dim=0), 
        values=Y.values().fill_(1.), 
        size=(n, c), dtype=torch.float).to_dense()

    if mask != None:
        onehot[~mask,:] = 0.
        return onehot

    return onehot


def square_p_loss(pred, edge_index):
    '''
    according to  https://github.com/yang-han/P-reg/
    '''
    sparse_adj = torch.sparse_coo_tensor(indices=edge_index, 
    values=torch.ones_like(edge_index[0]).to(torch.float), 
    size=(pred.shape[0], pred.shape[0]))
    conved_pred = torch.sparse.mm(sparse_adj, pred)
    square_loss = torch.norm(pred-conved_pred, p=2, dim=1)
    return square_loss

def mad_reg_loss(pred, edge_index):
    '''
    due to lacking the original code repository, we implement it on our own 
    '''
    # base = pred.norm(dim=1, keepdim=True, p=1).detach()
    # zero_idx = base.flatten() == 1.
    # base[zero_idx, 0] == 1.
    # pred = pred.div(base)
    D = 1. - pred.mm(pred.T)
    adj = torch.sparse_coo_tensor(indices=edge_index, 
    values=torch.ones_like(edge_index[0]).to(torch.float), 
    size=(pred.shape[0], pred.shape[0])).to_dense()
    rmt = 0.
    neb = 0.
    for i in range(6):
        adj = torch.mm(adj, adj)
        if i == 1:
            neb = adj
    high_neb = adj.nonzero().T
    rmt = 1. - torch.sparse_coo_tensor(indices=high_neb, 
    values=torch.ones_like(high_neb[0]).to(torch.float), 
    size=(pred.shape[0], pred.shape[0])).to_dense()

    mad_rmt = (D * rmt).mean()
    mad_neb = (D * neb).mean()

    return (mad_rmt - mad_neb)
