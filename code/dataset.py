import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.utils as pyg_utils
from sklearn.model_selection import KFold
from torch.utils import data

current_path = os.path.split(os.path.realpath(__file__))[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InteractionDataset(data.Dataset):
    def __init__(self, interactions, labels):
        self.interactions = interactions
        self.labels = labels

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        idx1 = self.interactions[index][0]
        idx2 = self.interactions[index][1]
        y = self.labels[index]
        return (idx1, idx2), y


class MGCAT_Dataset(object):
    def __init__(self, dataset_name='DB2', cv=5, fold=1, seed=2022):
        print(f'Loading {dataset_name} ...')

        # positive interaction
        positive_pairs = pd.read_csv(f'{current_path}/../data/{dataset_name}/positive_pairs.csv', sep='\t',
                                     header=None, names=['lncRNA', 'miRNA'])
        num_lncRNAs = len(set(positive_pairs['lncRNA']))  # 770   1663
        num_miRNAs = len(set(positive_pairs['miRNA']))  # 275    258
        num_nodes = num_lncRNAs + num_miRNAs

        # negative interaction
        negative_pairs = pd.read_csv(f'{current_path}/../data/{dataset_name}/negative_pairs.csv', sep='\t',
                                     header=None, names=['lncRNA', 'miRNA'])
        positive_pairs['miRNA'] += num_lncRNAs
        positive_pairs = positive_pairs.values
        negative_pairs['miRNA'] += num_lncRNAs
        negative_pairs = negative_pairs.values

        # 5-CV
        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        for idx, (train_index, test_index) in enumerate(kf.split(positive_pairs), start=1):
            if idx != fold:
                continue
            self.train_pos, self.test_pos = positive_pairs[train_index], positive_pairs[test_index]
            self.train_neg, self.test_neg = negative_pairs[train_index], negative_pairs[test_index]

        lncRNA_names = pd.read_csv(f'{current_path}/../data/{dataset_name}/lncRNA_idx.csv', header=None,
                                   sep='\t')[1].values.tolist()
        miRNA_names = pd.read_csv(f'{current_path}/../data/{dataset_name}/miRNA_idx.csv', header=None,
                                  sep='\t')[1].values.tolist()
        self.name2idx_dict = {name: idx for idx, name in enumerate(lncRNA_names + miRNA_names)}
        self.idx2name_dict = {idx: name for idx, name in enumerate(lncRNA_names + miRNA_names)}
        self.features = {}
        for rna in ['lncRNA', 'miRNA']:
            type_fea_dict = {}
            for feature_type in ['ctd', 'kmer', 'doc2vec', 'role2vec']:
                feature_df = pd.read_csv(
                    f'{current_path}/../data/{dataset_name}/features/{rna}_{feature_type}.dict', header=None,
                    index_col=0)
                feature_df.index = feature_df.index.map(self.name2idx_dict)
                feature_df.sort_index(ascending=True, inplace=True)
                type_fea_dict[feature_type] = torch.tensor(feature_df.values, dtype=torch.float32).to(device)
            self.features[rna] = type_fea_dict

        edge_index = torch.tensor(self.train_pos.T, dtype=torch.long)
        self.edge_index = pyg_utils.to_undirected(edge_index).to(device)

    def fetch_train_loader(self, batch_size=1024, shuffle=True):
        interactions = np.concatenate([self.train_pos, self.train_neg])
        labels = np.concatenate([np.ones(len(self.train_pos)), np.zeros(len(self.train_neg))])
        train_dataset = InteractionDataset(interactions, labels)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        return train_loader

    def fetch_test_loader(self, batch_size=1024, shuffle=False):
        interactions = np.concatenate([self.test_pos, self.test_neg])
        labels = np.concatenate([np.ones(len(self.test_pos)), np.zeros(len(self.test_neg))])
        test_dataset = InteractionDataset(interactions, labels)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        return test_loader


if __name__ == '__main__':
    MGCAT_Dataset()
