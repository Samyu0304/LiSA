from model import Ours

def parse_method_ours(args, dataset, n, c, d, device):
    if args.gnn == 'gcn':
        model = Ours(args, n, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Ours(args, n, c, d, 'sage', device).to(device)
    elif args.gnn == 'gcnii':
        model = Ours(args, n, c, d, 'gcnii', device).to(device)
    elif args.gnn == 'sggcn':
        model = Ours(args, n, c, d, 'sggcn', device).to(device)
    elif args.gnn == 'appnp':
        model = Ours(args, n, c, d, 'appnp', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='twitch-e') #twitch-e
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--device', type=int, default=5,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcnii_alpha', type=float, default=.1,
                        help='alpha for gcnii')
    parser.add_argument('--gcnii_lamda', type=float, default=1.0,
                        help='lambda for gcnii')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--warm_up', type=int, default=0,
                        help='validation label proportion')
    # for augmentation model's learning
    parser.add_argument('--K', type=int, default=3,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--inner_steps', type=int, default=10,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--kld_weight', type=float, default=0.1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--dist_weight', type=float, default=0.1,
                        help='steps for graph learner before one step for GNN')
    # for graph edit
    parser.add_argument('--num_sample', type=int, default=5,
                        help='num of samples for each node with graph edit')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=0.01,
                        help='learning rate for graph learner with graph edit')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='learning rate for graph learner with graph edit')

