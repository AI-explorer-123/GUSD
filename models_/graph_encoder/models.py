import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import RGCNConv, HGTConv, GPSConv, GATConv, RGATConv, TransformerConv, to_hetero
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from models_.model_utils import *
from datetime import datetime
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=256, layers=2, dropout=0.3):
        super().__init__()
        self.lrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.x
        x = self.lrelu(self.linear1(x))
        x = self.dropout(x)
        x = self.lrelu(self.linear2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SimpleHGN_conv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim=200, beta=0.05, final_layer=False):
        super(SimpleHGN_conv, self).__init__(aggr="add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, edge_index, edge_type, pre_alpha=None):

        node_emb = self.propagate(
            x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        if self.final:
            output = F.normalize(output, dim=1)

        return output, self.alpha.detach()

    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        alpha = self.leaky_relu(
            self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1, 1)
        return out

    def update(self, aggr_out):
        return aggr_out


class SimpleHGN(torch.nn.Module):
    def __init__(self, cfg, beta=0.05):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        num_relations = cfg.model.graph_encoder.num_rels

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.lrelu = nn.LeakyReLU()
        # self.lrelu = nn.GELU()
        self.text1 = SimpleHGN_conv(
            num_edge_type=num_relations, in_channels=hidden_channels, out_channels=hidden_channels, beta=beta)
        self.text2 = SimpleHGN_conv(num_edge_type=num_relations, in_channels=hidden_channels,
                                    out_channels=hidden_channels, beta=beta, final_layer=True)

    def forward(self, x, edge_index, edge_type):
        out = self.lrelu(self.linear1(x))
        out, alpha = self.text1(out, edge_index, edge_type)
        out = F.dropout(out, training=self.training, p=self.dropout)
        out, _ = self.text2(out, edge_index, edge_type, alpha)
        out = self.lrelu(self.linear2(out))
        return out


class RGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        num_relations = cfg.model.graph_encoder.num_rels

        if cfg.model.graph_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels)
        self.linear2 = nn.Linear(
            in_features=hidden_channels, out_features=out_channels)

        self.text1 = RGCNConv(in_channels=hidden_channels,
                              out_channels=hidden_channels, num_relations=num_relations)
        self.text2 = RGCNConv(in_channels=hidden_channels,
                              out_channels=hidden_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.activation(self.linear1(x))
        x = self.text1(x, edge_index, edge_type)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.text2(x, edge_index, edge_type)
        x = self.activation(self.linear2(x))
        return x


class GAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads

        if cfg.model.graph_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels)
        self.linear2 = nn.Linear(
            in_features=hidden_channels, out_features=out_channels)

        self.text1 = GATConv(in_channels=hidden_channels,
                             out_channels=hidden_channels // heads,
                             dropout=0.3,
                             heads=heads)
        self.text2 = GATConv(in_channels=hidden_channels,
                             out_channels=hidden_channels // heads,
                             dropout=0.3,
                             heads=heads)

    def forward(self, x, edge_index, edge_type):
        x = self.activation(self.linear1(x))
        x = self.text1(x, edge_index)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.text2(x, edge_index)
        x = self.activation(self.linear2(x))
        return x


class RGAT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads
        num_rels = cfg.model.graph_encoder.num_rels

        if cfg.model.graph_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels)
        self.linear2 = nn.Linear(
            in_features=hidden_channels, out_features=out_channels)

        self.text1 = RGATConv(in_channels=hidden_channels,
                              out_channels=hidden_channels // heads,
                              heads=heads,
                              num_relations=num_rels,
                              dropout=self.dropout)
        self.text2 = RGATConv(in_channels=hidden_channels,
                              out_channels=hidden_channels // heads,
                              heads=heads,
                              num_relations=num_rels,
                              dropout=self.dropout)

    def forward(self, x, edge_index, edge_type):
        x = self.activation(self.linear1(x))
        x = self.text1(x, edge_index, edge_type)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.text2(x, edge_index, edge_type)
        x = self.activation(self.linear2(x))
        return x


class GT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads
        num_rels = cfg.model.graph_encoder.num_rels

        if cfg.model.graph_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels)
        self.linear2 = nn.Linear(
            in_features=hidden_channels, out_features=out_channels)

        self.text1 = TransformerConv(in_channels=hidden_channels,
                                     out_channels=hidden_channels // heads,
                                     heads=heads,
                                     dropout=self.dropout)
        self.text2 = TransformerConv(in_channels=hidden_channels,
                                     out_channels=hidden_channels // heads,
                                     heads=heads,
                                     dropout=self.dropout)

    def forward(self, x, edge_index, edge_type):
        x = self.activation(self.linear1(x))
        x = self.text1(x, edge_index)
        x = F.dropout(x, training=self.training,  p=self.dropout)

        x = self.text2(x, edge_index)
        x = self.activation(self.linear2(x))
        return x


class Semantic(torch.nn.Module):
    def __init__(self, in_channels, out_dim, num_heads, num_relations):
        super(Semantic, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations

        self.Wd = torch.nn.ModuleList()
        for _ in range(num_heads):
            self.Wd.append(torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(out_dim, 1, bias=False),
            ))
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, H):
        output = torch.zeros(H.shape[0], H.shape[-1]).to(H.device)
        for d in range(self.num_heads):
            W = self.Wd[d](H).mean(0).squeeze()
            beta = self.softmax(W)
            for r in range(self.num_relations):
                output += (beta[r].item()) * H[:, r, :]
        output /= self.num_heads
        return output


class RGTLayer(torch.nn.Module):
    def __init__(self, in_channels, out_dim, num_heads, num_relations, dropout):
        super(RGTLayer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations

        self.layers = torch.nn.ModuleList()
        for _ in range(num_relations):
            self.layers.append(TransformerConv(in_channels=in_channels, out_channels=out_dim,
                                               heads=num_heads, dropout=dropout, concat=False))
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channels + out_dim, in_channels),
            torch.nn.Sigmoid()
        )
        self.tanh = torch.nn.Tanh()

        self.semantic = Semantic(
            in_channels, in_channels, num_heads, num_relations)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index_list):
        H = torch.zeros(x.shape[0], self.num_relations,
                        self.out_dim).to(x.device)  # H 5301*128
        for r in range(self.num_relations):
            U = self.layers[r](x, edge_index_list[r])
            Z = self.gate(torch.cat([U, x], dim=1))
            H[:, r, :] = self.tanh(U) * Z + x * (1 - Z)

        return self.semantic(transpose(H, self.num_relations))


class RGT(torch.nn.Module):
    def __init__(self, cfg, num_heads=4):
        super(RGT, self).__init__()

        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        num_relations = cfg.model.graph_encoder.num_rels

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.conv1 = RGTLayer(in_channels=hidden_channels, out_dim=hidden_channels,
                              num_heads=num_heads, num_relations=num_relations, dropout=self.dropout)
        self.conv2 = RGTLayer(in_channels=hidden_channels, out_dim=hidden_channels,
                              num_heads=num_heads, num_relations=num_relations, dropout=self.dropout)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        edge_index_list = to_edge_index_list(
            edge_index, edge_type, self.num_relations)
        x = self.lrelu(self.linear1(x))
        x = self.conv1(x, edge_index_list)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index_list)
        x = self.lrelu(self.linear2(x))
        return self.classifier(x)


class HGT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads

        metadata = (['user', 'movie', 'review'],
                    [('movie', 'commented_by', 'review'),
                     ('user', 'post', 'review'),
                     ('review', 'posted_by', 'user')])

        self.linear_relu1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU()
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(cfg.model.graph_encoder.layers):
            hgtconv = HGTConv(
                in_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata, heads=heads)
            self.convs.append(hgtconv)
        self.linear_relu2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        data.x_dict = {node_type: self.linear_relu1(
            x) for node_type, x in data.x_dict.items()}
        for hgtconv in self.convs:
            data.x_dict = hgtconv(data.collect('x'), data.edge_index_dict)
            data.x_dict = {key: F.dropout(value, training=self.training, p=self.dropout)
                      for key, value in data.x_dict.items()}
        data.x_dict = {node_type: self.linear_relu2(
            x) for node_type, x in data.x_dict.items()}
        return data.x_dict


class GPS(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        num_relations = cfg.model.graph_encoder.num_rels

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.lrelu = nn.LeakyReLU()
        self.text1 = SimpleHGN_conv(
            num_edge_type=num_relations, in_channels=hidden_channels, out_channels=hidden_channels, beta=0.05)
        self.text2 = SimpleHGN_conv(num_edge_type=num_relations, in_channels=hidden_channels,
                                    out_channels=hidden_channels, beta=0.05, final_layer=True)

        self.gps1 = GPSConv(channels=hidden_channels,
                            conv=self.text1,
                            heads=4,
                            dropout=self.dropout,
                            act='LeakyReLU',
                            attn_type='multihead',
                            attn_kwargs={'dropout': self.dropout})

        self.gps2 = GPSConv(channels=hidden_channels,
                            conv=self.text2,
                            heads=4,
                            dropout=self.dropout,
                            act='LeakyReLU',
                            attn_type='multihead',
                            attn_kwargs={'dropout': self.dropout})

    def forward(self, x, edge_index, edge_type):
        out = self.lrelu(self.linear1(x))
        out, alpha = self.gps1(out, edge_index, edge_type=edge_type)
        out = F.dropout(out, training=self.training, p=self.dropout)
        out, _ = self.gps2(
            out, edge_index, edge_type=edge_type, pre_alpha=alpha)
        out = self.lrelu(self.linear2(out))
        return out


class GAT_layer(nn.Module):
    def __init__(self, cfg, first_layer):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads
        self.first_layer = first_layer

        if self.first_layer == True:
            self.linear1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(in_features=in_channels, out_features=hidden_channels))
        self.linear2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=in_channels, out_features=hidden_channels))

        self.gat = GATConv(in_channels=hidden_channels,
                           out_channels=hidden_channels // heads,
                           dropout=0.3,
                           heads=heads,
                           add_self_loops=False)
        self.dropout = Dropout(p=0.3)

    def forward(self, x, edge_index, edge_type):
        if self.first_layer:
            x = self.linear1(x)
        x = self.gat(x, edge_index)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SGN_layer(nn.Module):
    def __init__(self, cfg, first_layer):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        self.dropout = cfg.model.graph_encoder.dropout
        self.first_layer = first_layer

        if self.first_layer == True:
            self.linear1 = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(in_features=in_channels, out_features=hidden_channels))
        self.linear2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=in_channels, out_features=hidden_channels))

        self.sgn = SimpleHGN_conv(in_channels=hidden_channels,
                                  out_channels=hidden_channels,
                                  num_edge_type=cfg.model.graph_encoder.num_rels,
                                  rel_dim=1024,
                                  final_layer=True)
        self.dropout2 = Dropout(p=self.dropout)
        self.dropout1 = Dropout(p=self.dropout)

    def forward(self, x, edge_index, edge_type):
        if self.first_layer:
            x = self.dropout1(self.linear1(x))
        x, _ = self.sgn(x, edge_index, edge_type)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class hetero_genreformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.metadata = (['user', 'movie', 'review'],
                         [('movie', 'commented_by', 'review'),
                          ('user', 'post', 'review'),
                          ('review', 'posted_by', 'user'),
                          ('movie', 'is_similar_to', 'movie')])
        in_channels = cfg.model.graph_encoder.in_channels
        # self.review_trm = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024,
        #                                                                                   nhead=4,
        #                                                                                   dropout=0.3,
        #                                                                                   batch_first=True),
        #                                                        num_layers=2)
        #                                 for _ in range(2)])
        self.genre_trm = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024,
                                                                                         nhead=4,
                                                                                         dropout=0.3,
                                                                                         batch_first=True),
                                                              num_layers=2)
                                        for _ in range(2)])
        self.gnn = nn.ModuleList([to_hetero(GAT_layer(cfg, first_layer=True),
                                            metadata=self.metadata,
                                            aggr='mean'),
                                  to_hetero(GAT_layer(cfg, first_layer=False),
                                            metadata=self.metadata,
                                            aggr='mean')])
        self.review_lin = nn.ModuleList(
            [nn.Linear(in_channels*2, in_channels) for _ in range(2)])
        self.movie_lin = nn.ModuleList(
            [nn.Linear(in_channels*2, in_channels) for _ in range(2)])
        self.activation = nn.LeakyReLU()

        # self.genre = nn.Parameter(torch.zeros(
        #     (cfg.genre_num, in_channels)))
        # self.genre = torch.zeros(
        #     (cfg.genre_num, cfg.model.graph_encoder.in_channels)).to(torch.device(self.device))

    def forward(self, x_dict, edge_index_dict, review_genre, movie_genre):
        x_dict = self.gnn[0](x_dict, edge_index_dict)
        all_genre, mask = genre_forward(x=torch.cat((x_dict['review'], x_dict['movie']), dim=0),
                                        genre_num=30,
                                        genres=torch.cat((review_genre, movie_genre), dim=0))

        # all_genre = torch.cat((self.genre.unsqueeze(1), all_genre), dim=1)
        # genres = self.review_trm[0](all_genre, src_key_padding_mask=mask)[
        #     :, 0, :].unsqueeze(0)
        genres = self.genre_trm[0](
            all_genre, src_key_padding_mask=mask).squeeze()

        review_genre_embedding = genre_backward(
            genres=review_genre, genre_feature=genres)
        movie_genre_embedding = genre_backward(
            genres=movie_genre, genre_feature=genres)

        x_dict['review'] = self.activation(self.review_lin[0](
            torch.cat((x_dict['review'], review_genre_embedding), dim=1)))
        x_dict['movie'] = self.activation(self.movie_lin[0](
            torch.cat((x_dict['movie'], movie_genre_embedding), dim=1)))

        x_dict = self.gnn[1](x_dict, edge_index_dict)

        all_genre, mask = genre_forward(x=torch.cat((x_dict['review'], x_dict['movie']), dim=0),
                                        genre_num=30,
                                        genres=torch.cat((review_genre, movie_genre), dim=0))

        # all_genre = torch.cat((self.genre.unsqueeze(1), all_genre), dim=1)
        # genres = self.review_trm[0](all_genre, src_key_padding_mask=mask)[
        #     :, 0, :].unsqueeze(0)
        genres = self.genre_trm[1](
            all_genre, src_key_padding_mask=mask).squeeze()

        review_genre_embedding = genre_backward(
            genres=review_genre, genre_feature=genres)
        movie_genre_embedding = genre_backward(
            genres=movie_genre, genre_feature=genres)

        x_dict['review'] = self.activation(self.review_lin[1](
            torch.cat((x_dict['review'], review_genre_embedding), dim=1)))
        x_dict['movie'] = self.activation(self.movie_lin[1](
            torch.cat((x_dict['movie'], movie_genre_embedding), dim=1)))

        return x_dict


class genreformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        in_channels = cfg.model.graph_encoder.in_channels
        # self.review_trm = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1024,
        #                                                                                   nhead=4,
        #                                                                                   dropout=0.3,
        #                                                                                   batch_first=True),
        #                                                        num_layers=2)
        #                                 for _ in range(2)])
        self.genre_trm = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=cfg.model.graph_encoder.out_channels,
                                                                                         nhead=4,
                                                                                         dropout=0.3,
                                                                                         batch_first=True),
                                                              num_layers=4)
                                        for _ in range(2)])

        if cfg.model.graph_encoder.genreformer_gnn == 'GAT':
            self.gnn = nn.ModuleList([GAT_layer(cfg, first_layer=True),
                                      GAT_layer(cfg, first_layer=False)])
        elif cfg.model.graph_encoder.genreformer_gnn == 'SimpleHGN':
            self.gnn = nn.ModuleList([SGN_layer(cfg, first_layer=True),
                                      SGN_layer(cfg, first_layer=False)])

        self.fusion_lin = nn.ModuleList(
            [nn.Linear(in_channels*2, in_channels) for _ in range(2)])
        self.activation = nn.LeakyReLU()

        # self.genre = nn.Parameter(torch.zeros(
        #     (cfg.genre_num, in_channels)))
        # self.genre = torch.zeros(
        #     (cfg.genre_num, cfg.model.graph_encoder.in_channels)).to(torch.device(self.device))

    def forward(self, x, edge_index, edge_type, genre, genre_mask):
        x = self.gnn[0](x, edge_index, edge_type)
        all_genre, mask = genre_forward(x=x[genre_mask],
                                        genre_num=30,
                                        genres=genre)

        # all_genre = torch.cat((self.genre.unsqueeze(1), all_genre), dim=1)
        # genres = self.review_trm[0](all_genre, src_key_padding_mask=mask)[
        #     :, 0, :].unsqueeze(0)
        genres = self.genre_trm[0](
            all_genre, src_key_padding_mask=mask).squeeze()

        genre_embedding = genre_backward(
            genres=genre, genre_feature=genres)

        x[genre_mask] = self.activation(self.fusion_lin[0](
            torch.cat((x[genre_mask], genre_embedding), dim=1)))

        x = self.gnn[1](x, edge_index, edge_type)
        all_genre, mask = genre_forward(x=x[genre_mask],
                                        genre_num=30,
                                        genres=genre)

        # all_genre = torch.cat((self.genre.unsqueeze(1), all_genre), dim=1)
        # genres = self.review_trm[0](all_genre, src_key_padding_mask=mask)[
        #     :, 0, :].unsqueeze(0)
        genres = self.genre_trm[1](
            all_genre, src_key_padding_mask=mask).squeeze()

        genre_embedding = genre_backward(
            genres=genre, genre_feature=genres)

        x[genre_mask] = self.activation(self.fusion_lin[1](
            torch.cat((x[genre_mask], genre_embedding), dim=1)))

        return x


class GATK_layer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads, hop_num, decay, activation, dropout):
        super().__init__()
        self.hop_num = hop_num
        self.dropout = dropout
        self.decay = decay
        if activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        self.decoder = nn.ModuleList([nn.Linear(
            in_features=hidden_channels,
            out_features=out_channels)
            for _ in range(hop_num)
        ])

        self.gat = nn.ModuleList([GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // heads,
            dropout=0.3,
            heads=heads)
            for _ in range(hop_num)
        ])

    def forward(self, x, edge_index_k_hops):
        out = []
        for k in range(self.hop_num):
            x_k = self.gat[k](x, edge_index_k_hops[k])
            x_k = F.dropout(x_k, training=self.training, p=self.dropout)
            x_k = self.decoder[k](x_k)
            x_k = self.activation(x_k)
            out.append(self.decay[k]*x_k)
        x = torch.sum(torch.stack(out), dim=0)
        return x


class GAT_KH(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers    
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        heads = cfg.model.graph_encoder.heads
        alpha = cfg.model.hop_decay
        self.hop_num = cfg.model.hop_num
        self.k_hop = k_hop_adj if cfg.model.hop_adj else k_hop_neighbors_with_min_distance
        self.ratio = cfg.model.ratio
        if alpha == 0.0:
            self.decay = nn.Parameter(torch.ones(self.hop_num) / self.hop_num)
        else:
            self.decay = [np.exp(-alpha * k) for k in range(self.hop_num)]
        if cfg.model.graph_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        
        self.linear1 = nn.Linear(
            in_features=in_channels, out_features=hidden_channels)
        self.gatk_layers = nn.ModuleList([
            GATK_layer(in_channels=in_channels,
                       out_channels=out_channels,
                       hidden_channels=hidden_channels,
                       heads=heads,
                       hop_num=self.hop_num,
                       decay=self.decay,
                       activation=cfg.model.graph_encoder.activation,
                       dropout=self.dropout
                       )
            for _ in range(self.layers)
        ])
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(self.layers)
        ])

    def forward(self, x, edge_index, edge_type, genre, genre_mask):
        edge_index_k_hops = self.k_hop(
            edge_index=edge_index, num_nodes=x.shape[0], k=self.hop_num, device=x.device, ratio=self.ratio)
        
        x = self.activation(self.linear1(x))
        residual = x
        for gat_layer, layernorm in zip(self.gatk_layers, self.layernorms):
            x = gat_layer(x,  edge_index_k_hops)
            x = layernorm(x)
            x = x + residual  # Apply residual connection
            residual = x
        return x


class K_Genreformer_layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        in_channels = cfg.model.graph_encoder.in_channels
        hidden_channels = cfg.model.graph_encoder.hidden_channels
        out_channels = cfg.model.graph_encoder.out_channels
        self.dropout = cfg.model.graph_encoder.dropout
        self.hop_num = cfg.model.hop_num
        self.ratio = cfg.model.ratio
        heads = cfg.model.graph_encoder.heads
        alpha = cfg.model.hop_decay
        if alpha == 0.0:
            self.decay = nn.Parameter(torch.ones(self.hop_num) / self.hop_num)
        else:
            self.decay = [np.exp(-alpha * k) for k in range(self.hop_num)]

        self.gnn = GATK_layer(in_channels=in_channels,
                              out_channels=out_channels,
                              hidden_channels=hidden_channels,
                              heads=heads,
                              hop_num=self.hop_num,
                              decay=self.decay,
                              activation=cfg.model.graph_encoder.activation,
                              dropout=self.dropout
                              )
        self.genre_trm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=cfg.model.graph_encoder.out_channels,
                                                                          nhead=4,
                                                                          dropout=0.3,
                                                                          batch_first=True),
                                               num_layers=4)
        self.pooling = cfg.model.graph_encoder.pooling
        self.fusion_lin = nn.Linear(in_channels*2, in_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x, edge_index_k_hops, genre, genre_mask):
        x = self.gnn(x, edge_index_k_hops)
        all_genre, mask = genre_forward(x=x[genre_mask],
                                        genre_num=30,
                                        genres=genre)
        genres = self.genre_trm(
            all_genre, src_key_padding_mask=mask).squeeze()

        genre_embedding = genre_backward(
            genres=genre, genre_feature=genres, pooling=self.pooling)

        x[genre_mask] = self.activation(self.fusion_lin(
            torch.cat((x[genre_mask], genre_embedding), dim=1)))
        return x


class K_Genreformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.graph_encoder.layers
        self.hop_num = cfg.model.hop_num
        self.k_hop = k_hop_adj if cfg.model.hop_adj else k_hop_neighbors_with_min_distance
        self.ratio = cfg.model.ratio
        
        self.in_channels = cfg.model.graph_encoder.in_channels
        self.hidden_channels = cfg.model.graph_encoder.hidden_channels
        self.out_channels = cfg.model.graph_encoder.out_channels
        
        self.linear = nn.Linear(
            in_features=self.in_channels, out_features=self.hidden_channels)
        self.activation = nn.LeakyReLU()

        self.k_genreformer_layers = nn.ModuleList(
            [K_Genreformer_layer(cfg) for _ in range(self.layers)])
        self.layernorms = nn.ModuleList([
            nn.LayerNorm(self.out_channels) for _ in range(self.layers)
        ])
        
    def forward(self, x, edge_index, edge_type, genre, genre_mask):
        edge_index_k_hops = self.k_hop(
            edge_index=edge_index, num_nodes=x.shape[0], k=self.hop_num, device=x.device, ratio=self.ratio)
            
        x = self.activation(self.linear(x))
        residual = x
        for k_genreformer_layer, layernorm in zip(self.k_genreformer_layers, self.layernorms):
            x = k_genreformer_layer(
                x, edge_index_k_hops, genre, genre_mask)
            x = layernorm(x)
            x = x + residual  # Apply residual connection
            residual = x
        return x
