from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset

from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv

import torch
from torch_geometric.utils import to_dense_batch

class TransformerFC(torch.nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim * 2, embed_dim),
            torch.nn.Dropout(dropout),
        )
        self.batch_norm = torch.nn.BatchNorm1d(embed_dim)
        self.batch_norm_dummy = torch.nn.BatchNorm1d(embed_dim)
        self.batch_norm_aggregate = torch.nn.BatchNorm1d(embed_dim)

        self.dropout_aggregate = torch.nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.dropout = dropout

    def forward(self, x_prior, x):
        x = self.dropout_aggregate(x)
        x = x_prior + x
        x = self.batch_norm_aggregate(x)
        x = self.mlp(x) + x
        return self.batch_norm(x)

class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout=0.0):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.fc = TransformerFC(embed_dim)

    def forward(self, x, batch):
        z, real_nodes = to_dense_batch(x, batch)
        z = self.attention(z, z, z, key_padding_mask=~real_nodes)[0][real_nodes]
        return self.fc(x, z)


class Task(Enum):
    NEIGHBORS_MATCH = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, depth, train_fraction):
        if self is Task.NEIGHBORS_MATCH:
            dataset = DictionaryLookupDataset(depth)
        else:
            dataset = None

        return dataset.generate_data(train_fraction)


class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    Transformer = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim, attention_dropout=0.0):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.Transformer:
            return TransformerLayer(in_dim, 4, attention_dropout)


class STOP(Enum):
    TRAIN = auto()
    TEST = auto()

    @staticmethod
    def from_string(s):
        try:
            return STOP[s]
        except KeyError:
            raise ValueError()


def one_hot(key, depth):
    return [1 if i == key else 0 for i in range(depth)]
