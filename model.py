import torch
import numpy as np
from torch import dropout, sparse
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import comb
from torch_geometric.nn.conv import MessagePassing, ChebConv, GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, add_self_loops, dropout_adj
import torch_sparse
from utils import get_gamma

class Bern_prop(MessagePassing):
    def __init__(self, nclass, K, relu=False, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.relu = relu
        self.temp = nn.Parameter(torch.Tensor(self.K+1)) # train the \theta_k
        self.bns1 = nn.ModuleList()
        self.bns2 = nn.ModuleList()
        for i in range(self.K + 1):
            self.bns1.append(nn.BatchNorm1d(nclass))
            self.bns1[-1].reset_parameters()
            self.bns2.append(nn.BatchNorm1d(nclass))
            self.bns2[-1].reset_parameters()

        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index):

        if self.relu:
            TEMP = F.relu(self.temp)
        else:
            TEMP = self.temp

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, normalization='sym', dtype=x.dtype, num_nodes=x.shape[0])
        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.shape[0])

        tmp=[]
        tmp.append(x)
        for i in range(self.K): # iterably aggregate top-k neighbors, save (2T-L)^K x successively
            # edge_index2, norm2 = dropedge(edge_index2, norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None) # loss sparsity?
            x = self.bns1[i](x)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K-i-1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            x = self.bns2[i](x)
            for j in range(i):
                # edge_index1, norm1 = dropedge(edge_index1, norm1)
                x = self.propagate(edge_index1,x=x,norm=norm1,size=None)
                x = self.bns2[i](x)
                # x = F.relu(x)
            out = out + (comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp.data.tolist())       
        
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    refering to https://github.com/jianhao2016/GPRGNN/
    '''

    def __init__(self, nclass, K, alpha=2, init='ppr', gamma=None, bias=True, relu=False, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.init = init
        self.alpha = alpha
        self.relu = relu
        self.bns = nn.ModuleList()
        for i in range(self.K + 1):
            self.bns.append(nn.BatchNorm1d(nclass))
            self.bns[-1].reset_parameters()

        assert init in ['sgc', 'ppr', 'nppr', 'random', 'ws']
        if init == 'sgc':
            # sgc-like
            TEMP = 0.0*np.ones(K+1)
            # TEMP = np.ones(K+1)
            TEMP[-1] = 1.0
        elif init == 'ppr':
            # ppr-like
            TEMP = alpha*(1-alpha)**np.arange(K+1, dtype=float)
            TEMP[-1] = (1-alpha)**K
        elif init == 'nppr':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif init == 'random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif init == 'ws':
            # Specify Gamma
            TEMP = gamma

        # TEMP = np.zeros(K+1)

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):

        if self.relu:
            TEMP = F.relu(self.temp)
        else:
            TEMP = self.temp 

        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = get_laplacian(edge_index, normalization='sym', num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = add_self_loops(edge_index, norm, fill_value=-1., num_nodes=x.size(0))

        hidden = x*TEMP[0] / (self.K+1)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.bns[k](x)
            gamma = TEMP[k+1] / (self.K+1)
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GCN_Net(torch.nn.Module):
    def __init__(self, ninput, nclass, args, pred=False):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(in_channels=ninput, out_channels=args.nhid)
        self.conv2 = GCNConv(in_channels=args.nhid, out_channels=nclass)
        self.bn = torch.nn.BatchNorm1d(args.nhid)
        
        self.dropout = args.dropout
        self.pred = pred

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.bn.reset_parameters()
        
    def forward(self, data):
        feat, edge_index = data.x, data.edge_index
        x = F.dropout(feat, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        # x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.pred:
            return x
        
        return F.log_softmax(x, dim=1)

class GCN_deep(torch.nn.Module):
    def __init__(self, ninput, nclass, args, pred=False):
        super(GCN_deep, self).__init__()
        self.K = args.K
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels=ninput, out_channels=args.nhid))
        for i in range(self.K - 2):
            self.convs.append(GCNConv(in_channels=args.nhid, out_channels=args.nhid))
        self.out = GCNConv(in_channels=args.nhid, out_channels=nclass)
        self.bn = torch.nn.BatchNorm1d(args.nhid)
        
        self.dropout = args.dropout
        self.pred = pred

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.out.reset_parameters()
        
        self.bn.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            # x = conv(x, dropout_adj(edge_index, p=0.2)[0])
            # x = self.bn(x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.out(x, dropout_adj(edge_index, p=0.2)[0])
        x = self.out(x, edge_index)
        if self.pred:
            return x
        
        return F.log_softmax(x, dim=1)

class SAGE_Net(torch.nn.Module):
    def __init__(self, ninput, nclass, args, pred=False):
        super(SAGE_Net, self).__init__()
        self.conv1 = SAGEConv(in_channels=ninput, out_channels=args.nhid)
        self.conv2 = SAGEConv(in_channels=args.nhid, out_channels=nclass)
        self.bn = torch.nn.BatchNorm1d(args.nhid)
        
        self.dropout = args.dropout
        self.pred = pred

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.bn.reset_parameters()
        
    def forward(self, data):
        feat, edge_index = data.x, data.edge_index
        x = F.dropout(feat, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        # x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.pred:
            return x
        
        return F.log_softmax(x, dim=1)

class GAT_Net(torch.nn.Module):
    def __init__(self, ninput, nclass, args, pred=False):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(in_channels=ninput, out_channels=args.nhid, heads=8, dropout=args.dropout)
        self.conv2 = GATConv(in_channels=args.nhid * 8, out_channels=nclass, heads=1, concat=False, dropout=args.dropout)
        self.bn = torch.nn.BatchNorm1d(args.nhid * 8)
        
        self.dropout = args.dropout
        self.pred = pred

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.bn.reset_parameters()
        
    def forward(self, data):
        feat, edge_index = data.x, data.edge_index
        x = F.dropout(feat, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        # x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        if self.pred:
            return x
        return F.log_softmax(x, dim=1)

class Cheb_prop(MessagePassing):

    def __init__(self, nclass, K, relu=False, **kwargs):
        super(Cheb_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.relu = relu
        self.bns = torch.nn.ModuleList()
        for i in range(self.K + 1):
            self.bns.append(torch.nn.BatchNorm1d(nclass))
        self.temp = nn.Parameter(torch.Tensor(self.K+1)) # train the \theta_k
        self.reset_parameters()

    def reset_parameters(self):
        init = torch.tensor([1 + 0.1 * i for i in range(self.K+1)])
        self.temp.data.fill_(0.)
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):

        if self.relu:
            TEMP = F.relu(self.temp)
        else:
            TEMP = self.temp #/ self.temp.norm()

        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = get_laplacian(edge_index, normalization='sym', num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = add_self_loops(edge_index, norm, fill_value=-1., num_nodes=x.size(0))

        tmp = [x]
        h = self.propagate(edge_index, x=x, norm=norm)
        # h = self.bns[0](h)
        tmp.append(h)

        hidden = tmp[0] * TEMP[0] + tmp[1] * TEMP[1]

        for k in range(2, self.K + 1):

            h = self.propagate(edge_index, x=tmp[k-1], norm=norm)
            # h = self.bns[k-1](h)
            h = 2 * h - tmp[k-2]
            tmp.append(h)
            hidden = hidden + TEMP[k] * tmp[k]

        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class Lagendre_prop(MessagePassing):
    '''
    in the explicit formulas
    '''
    def __init__(self, K, relu=False, **kwargs):
        super(Lagendre_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.relu = relu

        self.temp = nn.Parameter(torch.Tensor(self.K+1)) # train the \theta_k
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index):
        
        if self.relu:
            TEMP = F.relu(self.temp)
        else:
            TEMP = self.temp

        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = get_laplacian(edge_index, normalization='sym', num_nodes=x.size(0), dtype=x.dtype)

        edge_index1, norm1 = edge_index, norm
        edge_index2, norm2 = add_self_loops(edge_index, norm, fill_value=-2., num_nodes=x.shape[0])

        tmp = [x]
        for k in range(self.K):
            tmp.append(self.propagate(edge_index1, x=tmp[-1], norm=norm1)) 
        
        hidden = tmp[-1] * TEMP[-1] * (comb(self.K, 0)**2 / 2**(self.K))

        for k in range(self.K):
            h = tmp[self.K-k-1]
            h = self.propagate(edge_index2, x=h, norm=norm2, size=None)
            for i in range(k):
                h = self.propagate(edge_index2, x=h, norm=norm2, size=None)

            hidden = hidden +  (comb(self.K, k+1)**2 / 2**(self.K)) * TEMP[k] * h
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class Lagendre2_prop(MessagePassing):
    '''
    in a iterable way to generate the polynomials
    '''
    def __init__(self, K, relu=False, **kwargs):
        super(Lagendre2_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.relu = relu

        self.temp = nn.Parameter(torch.Tensor(self.K+1)) # train the \theta_k
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index):
        
        if self.relu:
            TEMP = F.relu(self.temp)
        else:
            TEMP = self.temp

        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        edge_index, norm = get_laplacian(edge_index, normalization='sym', num_nodes=x.size(0), dtype=x.dtype)

        edge_index1, norm1 = add_self_loops(edge_index, norm, fill_value=-1., num_nodes=x.size(0))

        tmp = [x]
        tmp.append(self.propagate(edge_index1, x=x, norm=norm1))

        hidden = tmp[0] * TEMP[0] + tmp[1] * TEMP[1]

        for k in range(2, self.K + 1):
            tmp.append(((2 * k - 1) * self.propagate(edge_index1, x=tmp[k-1], norm=norm1) - (k - 1) * tmp[k-2])/k)
            hidden = hidden + TEMP[k] * tmp[k]

        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

# not used
class PolyNet(nn.Module):
    def __init__(self, ninput, nclass, args):
        super(PolyNet, self).__init__()
        self.lin1 = nn.Linear(ninput, args.nhid)
        self.lin2 = nn.Linear(args.nhid, nclass)
        self.bn1 = nn.BatchNorm1d(args.nhid)
        self.net = args.net
        
        act_map = {
            'relu': F.relu,
            'leaky': F.leaky_relu,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh
        }

        prop_map = {
            'cheb': Cheb_prop,
            'gprgnn': GPR_prop,
            'bern': Bern_prop,
            'lag': Lagendre_prop,
            'lag2': Lagendre2_prop
        }

        if self.net != 'mlp':
            self.prop1 = prop_map[args.net](nclass, args.K, relu=args.relu)
            self.bn2 = nn.BatchNorm1d(nclass)
            self.reset_parameters()
        else: # mlp
            self.prop1 = None

        self.dprate = args.drop_prop
        self.dropout = args.dropout
        self.act = act_map[args.act]

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.bn1.reset_parameters()
        if self.net != 'mlp': 
            self.bn2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        if self.net == 'mlp':
            return F.log_softmax(x, dim=1)

        else:
            # x = self.bn2(x)
            # x = self.act(x)
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            
            return F.log_softmax(x, dim=1)
        
class MLP(torch.nn.Module):
    def __init__(self, ninput, nclass, args):
        super(MLP, self).__init__()

        self.lin1 = torch.nn.Linear(ninput, args.nhid)
        self.lin2 = torch.nn.Linear(args.nhid, nclass)
        self.bn = torch.nn.BatchNorm1d(args.nhid)

        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=-1)

class plain_gcn(torch.nn.Module):
    def __init__(self, ninput, nclass, args):
        super(plain_gcn, self).__init__()
        self.lin1 = torch.nn.Linear(ninput, args.nhid)
        self.lin2 = torch.nn.Linear(args.nhid, nclass)
        
        self.dropout = args.dropout
        

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
        
    def forward(self, filtered, x):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = torch.mm(filtered, x)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = torch.mm(filtered, x)
      
        return F.log_softmax(x, dim=1)
        
class Cheb_Net(torch.nn.Module):
    def __init__(self, ninput, nclass, nhid, K):
        super(Cheb_Net, self).__init__()
        self.conv1 = ChebConv(in_channels=ninput, out_channels=nhid, K=K)
        self.lin1 = torch.nn.Linear(nhid, nclass)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.lin1(x)
        return x






