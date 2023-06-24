import os.path as osp
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import pickle as pkl
import torch
import random
import numpy as np
from tqdm import tqdm
from mmcv import Config
from drugood.datasets import build_dataset
from torch_geometric.utils import dense_to_sparse
from ogb.graphproppred import Evaluator
from scipy.spatial.distance import cdist
from torch_geometric.data import InMemoryDataset, Data

def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data

class MNIST75sp(InMemoryDataset):
    splits = ['test', 'train']

    def __init__(self, root, mode='train', use_mean_px=True,
                 use_coord=True, node_gt_att_threshold=0,
                 transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        super(MNIST75sp, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['mnist_75sp_train.pkl', 'mnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return ['mnist_75sp_train.pt', 'mnist_75sp_test.pt']

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print("raw data of `{}` doesn't exist, please download from our github.".format(file))
                raise FileNotFoundError

    def process(self):

        data_file = 'mnist_75sp_%s.pkl' % self.mode
        with open(osp.join(self.raw_dir, data_file), 'rb') as f:
            self.labels, self.sp_data = pkl.load(f)

        self.use_mean_px = self.use_mean_px
        self.use_coord = self.use_coord
        self.n_samples = len(self.labels)
        self.img_size = 28
        self.node_gt_att_threshold = self.node_gt_att_threshold

        self.edge_indices, self.xs, self.edge_attrs, self.node_gt_atts, self.edge_gt_atts = [], [], [], [], []
        data_list = []
        for index, sample in enumerate(self.sp_data):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones(N_nodes, 1)  # dummy features

            # replicate features to make it possible to test on colored images
            x = np.pad(x, ((0, 0), (2, 0)), 'edge')
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

            node_gt_att = torch.LongTensor(node_gt_att).view(-1)
            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            data_list.append(
                Data(
                    x=torch.tensor(x),
                    y=torch.LongTensor([self.labels[index]]),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_gt_att=node_gt_att,
                    edge_gt_att=edge_gt_att,
                    name=f'MNISTSP-{self.mode}-{index}', idx=index
                )
            )
        idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])

class MotifShift(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(MotifShift, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('MotifShift_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['MotifShift_train.pt', 'MotifShift_val.pt', 'MotifShift_test.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'MotifShift_train.npy')):
            print("raw data of `MotifShift` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):

        # return maximal node feat
        max_feat = 0
        file_names = ['train', 'val', 'test']
        for file_name in file_names:
            idx = self.raw_file_names.index('{}.npy'.format(file_name))
            edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
                osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)

            for idx, (edge_index, y, ground_truth, z, p) in enumerate(
                    zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
                cur_max_feat = max(z)

                if cur_max_feat > max_feat:
                    max_feat = cur_max_feat

        max_feat += 1

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
            osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(
                zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1


            x = torch.zeros(node_idx.size(0), max_feat)
            index = [i for i in range(node_idx.size(0))]

            x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=p,
                        edge_gt_att=torch.LongTensor(ground_truth),
                        name=f'MotifShift-{self.mode}-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('MotifShift_{}.pt'.format(self.mode))
        print(self.processed_paths[idx])
        print(len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[idx])

class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'SPMotif_train.npy')):
            print("raw data of `SPMotif` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
            osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(
                zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=p,
                        edge_gt_att=torch.LongTensor(ground_truth),
                        name=f'SPMotif-{self.mode}-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        print(self.processed_paths[idx])
        print(len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[idx])


class DrugOOD(InMemoryDataset):

    def __init__(self, root, dataset, name, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        init from original DrugOOD dataset in the form of dgl.heterograph.DGLHeteroGraph
        """
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data(root, dataset, name, mode)

    def load_data(self, root, dataset, name, mode):
        data_path = osp.join(root, name + "_" + mode + ".pt")
        if not osp.exists(data_path):
            data_list = []
            # for data in dataset:
            for step, data in tqdm(enumerate(dataset), total=len(dataset), desc="Converting"):
                graph = data['input']
                y = data['gt_label']
                group = data['group']
                #assay_type = data['assay_type']

                edge_index = graph.edges()
                edge_attr = graph.edata['x']  #.long()
                node_attr = graph.ndata['x']  #.long()
                new_data = Data(edge_index=torch.stack(list(edge_index), dim=0),
                                edge_attr=edge_attr,
                                x=node_attr,
                                y=y,
                                group=group,
                                )#assay_type=assay_type
                data_list.append(new_data)
            torch.save(self.collate(data_list), data_path)

        self.data, self.slices = torch.load(data_path)







class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)

    if name in ['SPMotif-0.33','SPMotif-0.5','SPMotif-0.7','SPMotif-0.9','SPMotif-1.0']:
        datadir = '/data2/junchi.yu/graph-level/data'
        train_dataset = SPMotif(osp.join(datadir, name), mode='train')
        val_dataset = SPMotif(osp.join(datadir, name), mode='val')
        test_dataset = SPMotif(osp.join(datadir, name), mode='test')

        return [train_dataset, val_dataset, test_dataset]

    elif name in ['MotifShift-0.33','MotifShift-0.5', 'MotifShift-0.7']:
        datadir = '/data2/junchi.yu/graph-level/data'
        train_dataset = MotifShift(osp.join(datadir, name), mode='train')
        val_dataset = MotifShift(osp.join(datadir, name), mode='val')
        test_dataset = MotifShift(osp.join(datadir, name), mode='test')

        return [train_dataset, val_dataset, test_dataset]


    elif name in ['lbap_core_ic50_assay', 'lbap_core_ic50_size', 'lbap_core_ic50_scaffold',
                  'lbap_core_ec50_assay', 'lbap_core_ec50_size', 'lbap_core_ec50_scaffold']:
        cfg_path = './drugood_dataset/configs/dataloader_cfg/' + name + '.py'
        cfg = Config.fromfile(cfg_path)
        #print(cfg)
        root = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'DrugOOD')
        #print(root)
        train_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=name, mode="train")
        val_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_val), name=name, mode="ood_val")
        test_dataset = DrugOOD(root=root, dataset=build_dataset(cfg.data.ood_test), name=name, mode="ood_test")

        return [train_dataset, val_dataset, test_dataset]

    elif name == 'mnist':
        num_classes = 10
        n_train_data, n_val_data = 20000, 5000
        path = '/data2/junchi.yu/graph-level/data/MNIST'
        train_val = MNIST75sp(path, mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
        train_val = train_val[perm_idx]
        train_dataset, val_dataset = train_val[:n_train_data], train_val[-n_val_data:]
        test_dataset = MNIST75sp(path, mode='test')


        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        return [train_dataset, val_dataset, test_dataset]


    else:
        path = '/data2/junchi.yu/graph-level/data/' + name
        dataset = TUDataset(path, name, cleaned=cleaned)
        dataset.data.edge_attr = None
        if dataset.data.x is None:
            max_degree = 0
            degs = [dataset.data.x is None]
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)
        if not sparse:
            num_nodes = max_num_nodes = 0
            for data in dataset:
                num_nodes += data.num_nodes
                max_num_nodes = max(data.num_nodes, max_num_nodes)
            num_nodes = max_num_nodes

            if dataset.transform is None:

                dataset.transform = T.ToDense(num_nodes)
            else:
                dataset.transform = T.Compose(
                    [dataset.transform, T.ToDense(num_nodes)])
        return dataset

if __name__ == "__main__":
    '''
    dataset = get_dataset(name='COLLAB')
    all_count = 0
    big_count = 0
    train_idx = []
    val_idx = []
    test_idx = []
    for idx, data in enumerate(dataset):
        size = data.x.size()[0]
        if size <= 50:
            train_idx.append(idx)
        elif size <= 150:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    print(len(train_idx), len(val_idx), len(test_idx))
    
    from torch_geometric.data import DataLoader
    datadir = './data/'
    bias = 0.7
    batch_size = 10
    train_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(val_dataset.num_classes, val_dataset.num_features)
    '''
    from torch_geometric.data import DataLoader
    dataloader = MNIST75sp(root='/data2/junchi.yu/graph-level/data/MNIST', mode = 'train')
    data = DataLoader(dataloader, batch_size=4, shuffle=True)
    print(data)







