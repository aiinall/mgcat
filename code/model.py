import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import NodeAttLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MGCAT_Model(nn.Module):
    def __init__(self, raw_features, edge_index, embedding_size=128):
        super(MGCAT_Model, self).__init__()
        self.raw_features = raw_features
        self.edge_index = edge_index

        self.lncRNA_feature_proj = {}
        self.miRNA_feature_proj = {}
        self.feature_types = ['ctd', 'kmer', 'doc2vec', 'role2vec']
        for f in self.feature_types:
            lncRNA_feature_dim = raw_features['lncRNA'][f].shape[1]
            miRNA_feature_dim = raw_features['miRNA'][f].shape[1]
            self.lncRNA_feature_proj[f] = nn.Linear(lncRNA_feature_dim, embedding_size).to(device)
            self.miRNA_feature_proj[f] = nn.Linear(miRNA_feature_dim, embedding_size).to(device)

        self.conv1 = NodeAttLayer(embedding_size, embedding_size, heads=1)
        self.norm1 = nn.BatchNorm1d(embedding_size)
        self.conv2 = NodeAttLayer(embedding_size, embedding_size, heads=1)
        self.norm2 = nn.BatchNorm1d(embedding_size)
        self.conv3 = NodeAttLayer(embedding_size, embedding_size, heads=1)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(embedding_size * 2),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, 1),
            nn.Sigmoid(),
        )
        self.criterion = torch.nn.BCELoss(reduction='sum')

        self.lncRNA_att = nn.Linear(embedding_size, 1)
        self.miRNA_att = nn.Linear(embedding_size, 1)
        self.layer_att = nn.Linear(embedding_size, 1)

    def project(self):
        projected_lncRNA_features = []
        projected_miRNA_features = []
        for f in self.feature_types:
            projected_lncRNA_feature = self.lncRNA_feature_proj[f](self.raw_features['lncRNA'][f])
            projected_lncRNA_features.append(projected_lncRNA_feature)
            projected_miRNA_feature = self.miRNA_feature_proj[f](self.raw_features['miRNA'][f])
            projected_miRNA_features.append(projected_miRNA_feature)
        return projected_lncRNA_features, projected_miRNA_features

    def forward(self, idx, lbl=None):
        projected_lncRNA_features, projected_miRNA_features = self.project()

        lncRNA_features = torch.stack(projected_lncRNA_features, dim=1)
        lncRNA_w = self.lncRNA_att(lncRNA_features)
        lncRNA_w = lncRNA_w.squeeze()
        lncRNA_w = torch.sigmoid(lncRNA_w)
        lncRNA_w = lncRNA_w.unsqueeze(1)
        lncRNA_features = torch.matmul(lncRNA_w, lncRNA_features).squeeze()
        miRNA_features = torch.stack(projected_miRNA_features, dim=1)
        miRNA_w = self.miRNA_att(miRNA_features)
        miRNA_w = miRNA_w.squeeze()
        miRNA_w = torch.sigmoid(miRNA_w)
        miRNA_w = miRNA_w.unsqueeze(1)
        miRNA_features = torch.matmul(miRNA_w, miRNA_features).squeeze()

        x0 = torch.cat((lncRNA_features, miRNA_features), dim=0)
        x_list = [x0]
        x0 = F.dropout(x0, p=0.1)
        x1 = self.conv1(x0, self.edge_index)
        x_list.append(x1)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, self.edge_index)
        x_list.append(x2)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, self.edge_index)
        x_list.append(x3)

        x_features = torch.stack(x_list, dim=1)
        x_w = self.layer_att(x_features)
        x_w = x_w.squeeze()
        x_w = torch.sigmoid(x_w)
        x_w = x_w.unsqueeze(1)
        z = torch.matmul(x_w, x_features).squeeze()

        u_feature = z[idx[0]]
        v_feature = z[idx[1]]
        uv_feature = torch.cat((u_feature, v_feature), dim=1)
        out = torch.squeeze(self.mlp(uv_feature))

        if lbl is not None:
            loss_train = self.criterion(out, lbl.float())
            return out, loss_train
        else:
            return out, None
