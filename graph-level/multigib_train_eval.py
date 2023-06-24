import time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset_name, dataset, model, generators, runs, epochs, batch_size,
                                  lr, lr_a, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, inner_loop, kld_weight, dist_weight, device, rocauc, joint, logger=None):

    val_losses, accs, durations = [], [], []

    if dataset_name in ['SPMotif-0.33', 'SPMotif-0.5', 'SPMotif-0.7', 'SPMotif-0.9', 'SPMotif-1.0',
                        'lbap_core_ic50_assay', 'lbap_core_ic50_size', 'lbap_core_ic50_scaffold',
                        'lbap_core_ec50_assay', 'lbap_core_ec50_size', 'lbap_core_ec50_scaffold', 'mnist']:
        train_dataset, val_dataset, test_dataset = dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'DD':

        train_idx = []
        val_idx = []
        test_idx = []
        for idx, data in enumerate(dataset):
            size = data.x.size()[0]
            if size <= 200:
                train_idx.append(idx)
            elif size <= 300:
                val_idx.append(idx)
            else:
                test_idx.append(idx)
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    elif dataset_name == 'MUTAG':
        train_idx = []
        val_idx = []
        test_idx = []

        for idx, data in enumerate(dataset):
            size = data.x.size()[0]
            if size <= 15:
                train_idx.append(idx)
            elif size <= 20:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    for run in range(runs):

        model.reset_parameters()
        optimizer_model = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        generators_params = []
        for generator in generators:
            generator.reset_parameters()
            generators_params.append(generator.parameters())

        # initialize optimizer
        optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params), lr=lr_a)


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        cnt, last_val_acc = 0, 0

        for epoch in range(1, epochs + 1):
            if joint:
                train_loss = train_joint(model, generators, optimizer_model, optimizer_generator,
                                         train_loader, kld_weight, dist_weight, inner_loop, device)
            else:
                train_loss = train_edge(model, generators, optimizer_model, optimizer_generator,
                                        train_loader, kld_weight, dist_weight, inner_loop, device)

            if train_loss != train_loss:
                print('NaN')
                continue

            #print(train_loss)

            if rocauc:
                val_loss = eval_rocauc(model, val_loader, device)#eval_acc(model, val_loader, device)
                test_acc = eval_rocauc(model, test_loader, device)
            else:
                val_loss = eval_acc(model, val_loader, device)
                test_acc = eval_acc(model, test_loader, device)

            val_losses.append(val_loss)
            accs.append(test_acc)#eval_acc(model, test_loader, device)
            eval_info = {
                'run': run,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            #print(eval_info)

            if logger is not None:
                logger(eval_info)
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer_model.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(runs, epochs), acc.view(runs, epochs)
    loss, argmin = loss.max(dim=1)
    _, argmax = acc.max(dim=1)
    oracle_acc = acc[torch.arange(runs, dtype=torch.long), argmax]
    val_acc = acc[torch.arange(runs, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    oracle_acc_mean = oracle_acc.mean().item()
    oracle_acc_std = oracle_acc.std().item()

    val_acc_mean = val_acc.mean().item()
    val_acc_std = val_acc.std().item()
    duration_mean = duration.mean().item()
    print(
        'Val Loss: {:.4f}, oracle Test Accuracy: {:.3f} ± {:.3f}, Val Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
        format(loss_mean, oracle_acc_mean, oracle_acc_std, val_acc_mean, val_acc_std, duration_mean))

    return loss_mean, oracle_acc_mean, oracle_acc_std, val_acc_mean, val_acc_std


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train_joint(model, generators, optimizer_model, optimizer_generator, loader, kld_weight, dist_weight, inner_loop, device):
    model.train()
    num_generators = len(generators)

    for data in loader:

        data = data.to(device)

        for j in range(0, inner_loop):

            optimizer_generator.zero_grad()
            loss_array = []
            sqrt_loss_array = []
            kld_array = []
            out_og, out_embs = model(data)
            loss_og = F.nll_loss(out_og, data.y.view(-1))
            loss_array.append(loss_og.view(-1))

            for k in range(0, num_generators):
                generator = generators[k]

                kld_loss, node_mask, edge_mask = generator(out_embs, data.edge_index, data.batch)
                kld_array.append(kld_loss.view(-1))

                set_masks(edge_mask, model)
                out_local, _ = model(data, mask = node_mask)
                clear_masks(model)

                loss_local = F.nll_loss(out_local, data.y.view(-1))


                loss_array.append(loss_local.view(-1))
                sqrt_loss_array.append(torch.sqrt(loss_local).view(-1))



            Loss = torch.cat(loss_array, dim=0)
            sqrt_loss = torch.cat(sqrt_loss_array, dim=0)
            _, Mean = torch.var_mean(Loss)
            Var, _ = torch.var_mean(sqrt_loss)
            kld_loss = torch.cat(kld_array, dim=0)
            _, kld_loss = torch.var_mean(kld_loss)

            optimizer_generator.zero_grad()
            loss_generator = Mean + kld_weight * kld_loss - dist_weight * Var
            loss_generator.backward()
            optimizer_generator.step()

        loss_array = []

        out_og, out_embs = model(data)
        loss_og = F.nll_loss(out_og, data.y.view(-1))
        loss_array.append(loss_og.view(-1))

        for k in range(0, num_generators):
            generator = generators[k]

            kld_loss, node_mask, edge_mask = generator(out_embs, data.edge_index, data.batch)

            set_masks(edge_mask, model)
            out_local, _ = model(data, node_mask)
            clear_masks(model)

            loss_local = F.nll_loss(out_local, data.y.view(-1))
            loss_array.append(loss_local.view(-1))

        Loss = torch.cat(loss_array, dim=0)
        Var, Mean = torch.var_mean(Loss)



        optimizer_model.zero_grad()
        loss_classifier = Mean + dist_weight * Var
        #print(Mean, Var)
        loss_classifier.backward()
        optimizer_model.step()

    return loss_classifier


def train_edge(model, generators, optimizer_model, optimizer_generator, loader, kld_weight, dist_weight, inner_loop, device):
    model.train()
    num_generators = len(generators)

    for data in loader:

        data = data.to(device)

        for j in range(0, inner_loop):

            optimizer_generator.zero_grad()
            loss_array = []
            sqrt_loss_array = []
            kld_array = []
            out_og, out_embs = model(data)
            loss_og = F.nll_loss(out_og, data.y.view(-1))
            loss_array.append(loss_og.view(-1))

            for k in range(0, num_generators):
                generator = generators[k]

                kld_loss, mask = generator(out_embs, data.edge_index)
                kld_array.append(kld_loss.view(-1))
                set_masks(mask, model)
                out_local, _ = model(data)
                loss_local = F.nll_loss(out_local, data.y.view(-1))
                loss_array.append(loss_local.view(-1))
                sqrt_loss_array.append(torch.sqrt(loss_local).view(-1))
                clear_masks(model)

            Loss = torch.cat(loss_array, dim=0)
            sqrt_loss = torch.cat(sqrt_loss_array, dim=0)
            _, Mean = torch.var_mean(Loss)
            Var, _ = torch.var_mean(sqrt_loss)
            kld_loss = torch.cat(kld_array, dim=0)
            _, kld_loss = torch.var_mean(kld_loss)

            optimizer_generator.zero_grad()
            loss_generator = Mean + kld_weight * kld_loss - dist_weight * Var
            loss_generator.backward()
            optimizer_generator.step()

        loss_array = []
        out_og, out_embs = model(data)
        loss_og = F.nll_loss(out_og, data.y.view(-1))
        loss_array.append(loss_og.view(-1))

        for k in range(0, num_generators):
            generator = generators[k]

            kld_loss, mask = generator(out_embs, data.edge_index)
            set_masks(mask, model)
            out_local, _ = model(data)
            loss_local = F.nll_loss(out_local, data.y.view(-1))
            loss_array.append(loss_local.view(-1))
            clear_masks(model)

        Loss = torch.cat(loss_array, dim=0)
        Var, Mean = torch.var_mean(Loss)

        optimizer_model.zero_grad()
        loss_classifier = Mean + dist_weight * Var
        #print(Mean, Var)
        loss_classifier.backward()
        optimizer_model.step()

    return loss_classifier

def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred,_ = model(data)
            pred = pred.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_rocauc(model, loader, device):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""

    model.eval()

    y_true, y_pred = [], []
    rocauc_list = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred,_ = model(data)
            label = data.y
            y_true.append(label.view(-1))
            y_pred.append(pred)

    y_true = torch.cat(y_true, dim = 0)
    y_true = y_true.unsqueeze(dim = -1)
    y_pred = torch.cat(y_pred, dim = 0)
    #print(y_true.size(), y_pred.size())

    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

def eval_loss(model, loader, device):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,_ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


