from itertools import product

import argparse
from datasets import get_dataset
from multigib_train_eval import cross_validation_with_val_set
import torch

from multigib_gcn import MultiGCN, EdgeGenerator, JointGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)#default = 100
parser.add_argument('--batch_size', type=int, default=128)#default = 128
parser.add_argument('--lr', type=float, default=0.001)#motif: 0.001
parser.add_argument('--lr_a', type=float, default=0.001)#motif: 0.001
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--net', type=int, default=1)
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--inner_loop', type=int, default=20)
parser.add_argument('--kld_weight', type=float, default=0.1)
parser.add_argument('--dist_weight', type=float, default=0.1)
parser.add_argument('--rocauc', action = 'store_true')
parser.add_argument('--joint', action = 'store_true')
parser.add_argument('--hiddens', type=int, default=64)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--cuda', type=int, default=2)
args = parser.parse_args()

layers = [args.layers]
hiddens = [args.hiddens]
datasets = [args.dataset]#['SPMotif-0.33','SPMotif-0.5','SPMotif-0.7','SPMotif-0.9','SPMotif-1.0']  # , 'COLLAB']DDK
nets = [MultiGCN]
device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse= True)
        if dataset_name in ['SPMotif-0.33', 'SPMotif-0.5', 'SPMotif-0.7', 'SPMotif-0.9', 'SPMotif-1.0',
                            'lbap_core_ic50_assay', 'lbap_core_ic50_size', 'lbap_core_ic50_scaffold',
                            'lbap_core_ec50_assay', 'lbap_core_ec50_size', 'lbap_core_ec50_scaffold', 'mnist']:
            model = Net(dataset[0], num_layers, hidden, args.joint).to(device)
        else:
            model = Net(dataset, num_layers, hidden, args.joint).to(device)
        generators = []
        for i in range(args.K):
            if args.joint:
                generators.append(JointGenerator(hidden_size=hidden, device = device).to(device))
            else:
                generators.append(EdgeGenerator(hidden_size=hidden, device=device).to(device))


        loss, oracle_acc, oracle_std, val_acc, val_std = cross_validation_with_val_set(
            dataset_name = dataset_name,
            dataset=dataset,
            model=model,
            generators = generators,
            runs=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_a = args.lr_a,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            inner_loop=args.inner_loop,
            kld_weight=args.kld_weight,
            dist_weight=args.dist_weight,
            device=device,
            rocauc = args.rocauc,
            joint = args.joint,
            logger=None,
        )
        if loss < best_result[0]:
            best_result = (loss, oracle_acc, oracle_std, val_acc, val_std)

    desc = '{:.3f} +- {:.3f}, {:.3f} +- {:.3f}'.format(best_result[1], best_result[2], best_result[3], best_result[4])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
