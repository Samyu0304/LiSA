import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from itertools import chain
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, evaluate_whole_graph_multi, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, our_forward, load_fixed_splits
from parse import parse_method_ours,  parser_add_main_args
from model import Ours, Graph_Editer

import warnings
warnings.filterwarnings("ignore")

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, ratio=None, sub_dataset=None):
    ### Load and preprocess data ###
    if dataset == 'twitch-e':
        dataset = load_nc_dataset('twitch-e', sub_dataset)
    elif dataset == 'fb100':
        dataset = load_nc_dataset('fb100', sub_dataset)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']
    return dataset

if args.dataset == 'twitch-e':
    twitch_sub_name = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']
    tr_sub, val_sub, te_subs = ['DE'], ['ENGB'], ['ES', 'FR', 'PTBR', 'RU', 'TW']
    datasets_tr = [get_dataset(dataset='twitch-e', sub_dataset=tr_sub[0])]
    datasets_val = [get_dataset(dataset='twitch-e', sub_dataset=val_sub[0])]
    datasets_te = [get_dataset(dataset='twitch-e', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
elif args.dataset == 'fb1001':
    tr_subs, val_subs, te_subs = ['Johns Hopkins55', 'Caltech36', 'Amherst41'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
    datasets_tr = [get_dataset(dataset='fb100', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='fb100', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='fb100', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
elif args.dataset == 'fb1002':
    tr_subs, val_subs, te_subs = ['Bingham82', 'Duke14', 'Princeton12'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
    datasets_tr = [get_dataset(dataset='fb100', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='fb100', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='fb100', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
elif args.dataset == 'fb1003':
    tr_subs, val_subs, te_subs = ['WashU32', 'Brandeis99', 'Carnegie49'], ['Cornell5', 'Yale4'], ['Penn94', 'Brown11', 'Texas80']
    datasets_tr = [get_dataset(dataset='fb100', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='fb100', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='fb100', sub_dataset=te_subs[i]) for i in range(len(te_subs))]

else:
    raise ValueError('Invalid dataname')

#if args.dataset in ['fb1001', 'fb1002', 'fb1003']:
dataset_tr = datasets_tr[0]
dataset_val = datasets_val[0]
print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

### Load method ###
model = Ours(args,  dataset_tr.c, dataset_tr.d, args.gnn, device)
# using rocauc as the eval function
criterion = nn.BCEWithLogitsLoss()
if args.dataset in ['fb1001', 'fb1002', 'fb1003']:
    eval_func = eval_acc
else:
    eval_func = eval_rocauc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)
print('DATASET:', args.dataset)

generators = []
for dataset in datasets_tr:
    edges = dataset.graph['edge_index']
    num_edges = edges.size()[1]
    #print(num_edges)
    generators.append(Graph_Editer(args.K, num_edges, args.device))



### Training loop ###
for run in range(args.runs):
    model.reset_parameters()

    generators_params = []
    for generator in generators:
        generator.reset_parameters()
        generators_params.append(generator.parameters())

    #initialize optimizer
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params),
                                            lr=args.lr_a, weight_decay=args.weight_decay)

    #torch.autograd.set_detect_anomaly(True)
    best_val = float('-inf')

    val_history = []

    for epoch in range(args.epochs):
        model.train()

        for dataset_ind in range(len(datasets_tr)):
            dataset_tr = datasets_tr[dataset_ind]
            generators_per_dataset = generators[dataset_ind]

            # back prop generator

            for inner_steps in range(args.inner_steps):

                loss_array = []
                loss_var_array = []
                kld_array = []
                loss_og = model(dataset_tr, criterion)
                loss_array.append(loss_og.view(-1))

                for k in range(0, args.K):
                    mask_per_view, kld_loss = generators_per_dataset(k)
                    kld_array.append(kld_loss.view(-1))
                    loss_local = model(dataset_tr, criterion, mask_per_view)
                    loss_array.append(loss_local.view(-1))
                    loss_var_array.append(torch.sqrt(loss_local).view(-1))

                Loss = torch.cat(loss_array, dim=0)
                Loss_var = torch.cat(loss_var_array, dim=0)
                _, Mean = torch.var_mean(Loss)
                Var, _ = torch.var_mean(Loss_var)
                kld_loss = torch.cat(kld_array, dim=0)
                _, kld_loss = torch.var_mean(kld_loss)

                optimizer_generator.zero_grad()
                loss_generator = Mean + args.kld_weight * kld_loss - args.dist_weight * Var
                loss_generator.backward()
                optimizer_generator.step()

            loss_array = []
            loss_og = model(dataset_tr, criterion)
            loss_array.append(loss_og.view(-1))

            for k in range(0, args.K):
                mask_per_view = generators_per_dataset.sample(k)
                loss_local = model(dataset_tr, criterion, mask_per_view)
                loss_array.append(loss_local.view(-1))

            Loss = torch.cat(loss_array, dim=0)
            Var, Mean = torch.var_mean(Loss)
            optimizer_model.zero_grad()
            loss_classifier = Mean + Var

            loss_classifier.backward()
            optimizer_model.step()


        if args.dataset == 'twitch-e':
            accs, test_outs = evaluate_whole_graph(args, model, datasets_tr[0], datasets_val[0], datasets_te, eval_func)
        elif args.dataset in ['fb1001', 'fb1002','fb1003']:
            accs, test_outs = evaluate_whole_graph_multi(args, model, datasets_tr, datasets_val, datasets_te, eval_func)

        logger.add_result(run, accs)

        #early stop
        val_history.append(accs[1])
        if args.early_stopping > 0 and epoch > args.warm_up:
            tmp = torch.tensor(val_history[-(args.early_stopping + 1):-1])
            if accs[1] < tmp.mean().item():
                break

        if epoch % args.display_step == 0:

            print(f'Epoch: {epoch:02d}, '
                    f'Mean Loss: {Mean:.4f}, '
                    f'Var Loss: {Var:.4f}, '
                    f'Kld Loss: {kld_loss:.4f}, '
                    f'Train: {100 * accs[0]:.2f}%, '
                    f'Valid: {100 * accs[1]:.2f}%, ')
            test_info = ''
            for test_acc in accs[2:]:
                test_info += f'Test: {100 * test_acc:.2f}% '
            print(test_info)

    logger.print_statistics(run)


### Save results ###
results = logger.print_statistics()
filename = f'./our_results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    # sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    log = f"{args.method}," + f"{args.gnn},"
    for i in range(results.shape[1]):
        r = results[:, i]
        log += f"{r.mean():.3f} ± {r.std():.3f},"
    write_obj.write(log + f"\n")
