import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse


from nets import *

class Ours(nn.Module):
    def __init__(self, args, c, d, gnn, device, dataset=None):
        super(Ours, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)
        elif gnn == 'sage':
            self.gnn = OurSAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout).to(device)
        elif gnn == 'gcnii':
            self.gnn = GCN2Net(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        lam = args.gcnii_lamda,
                        alp = args.gcnii_alpha).to(device)

        self.device = device
        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def energy_scores(self, logits):

        exp_logits = torch.exp(logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1)

        return torch.log(sum_exp_logits)


    def forward(self, data, criterion, mask=None):

        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index, mask)

        #out = torch.clamp(out, -10, 10)
        if self.args.dataset == 'elliptic':
            loss = self.sup_loss(y[data.mask], out[data.mask], criterion)
        else:
            loss = self.sup_loss(y, out, criterion)

        #scores = self.energy_scores(out)
        return loss

    def inference(self, data, mask=None):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index, mask)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb1001', 'fb1002','fb1003','elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss


class Graph_Editer(nn.Module):
    def __init__(self, K, edge_num, device):
        super(Graph_Editer, self).__init__()
        self.K = K
        self.edge_num = edge_num
        self.S = 0.1 #0.1 for face book
        self.sample_size = int(self.S * edge_num)
        self.B = nn.Parameter(torch.FloatTensor(K, edge_num))

        #print(self.B)
        self.epsilon = 0.000001
        self.temperature = 1.0

        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + 0.00000001) + neg * torch.log(neg/0.5 + 0.000000001))

        return kld_loss

    def forward(self, k):
        #return a KL-like loss term to control the information flow
        Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        #mask = mask.unsqueeze(dim = 0)
        #cat_mask = torch.cat([mask,1-mask],dim = 0)
        sample_mask = self.straight_through(mask)#F.gumbel_softmax(cat_mask, tau=1, dim = 0, hard=False)
        kld_loss = self.kld(mask)
        #print(sample_mask[0])
        #print(kld_loss)

        return sample_mask, kld_loss

    def sample(self, k):
        Bk = self.B[k]
        mask = torch.clamp(Bk, -10, 10).to(self.device)
        mask = torch.sigmoid(mask)
        mask = self.straight_through(mask)#torch.sigmoid(mask)

        #mask = mask.unsqueeze(dim=0)
        #cat_mask = torch.cat([mask, 1 - mask], dim=0)
        #sample_mask = F.gumbel_softmax(cat_mask, tau=1, dim=0, hard=False)
        #print(sample_mask)

        return mask
    #straight-through sampling

    def straight_through(self, mask):
        _, idx = torch.topk(mask, self.sample_size)
        sample_mask = torch.zeros_like(mask).to(self.device)
        sample_mask[idx]=1

        return sample_mask + mask - mask.detach()






if __name__=='__main__':
    import torch.functional as F
    import torch

    a = torch.Tensor([1,2,3,4])

