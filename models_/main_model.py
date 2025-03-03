import torch
import torch.nn as nn
from models_.new_moe import NewMoE
from models_.model_utils import split, hetero_split, get_graph_encoder
from soft_mixture_of_experts.transformer import SoftMoEEncoder, SoftMoEEncoderLayer
from soft_mixture_of_experts.soft_moe import SoftMoE
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE
from models_.moe import MoE


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = channels

        self.classifier = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, cfg.model.num_classes)
        )

    def forward(self, data):
        movie_graph, user_graph, review_graph = split(
            data.x, data.movie_map, data.user_map, data.review_map)
        out = self.classifier(review_graph)
        return out


class GNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.model.graph_encoder.out_channels
        graph_encoder = get_graph_encoder(cfg.model.graph_encoder.name)
        self.graph_encoder = graph_encoder(cfg)
        self.classifier = nn.Linear(channels, cfg.model.num_classes)

    def forward(self, data):
        graph_feature, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        graph_feature = self.graph_encoder(
            graph_feature, edge_index, edge_type)
        movie_graph, user_graph, review_graph = split(
            graph_feature, data.movie_map, data.user_map, data.review_map)
        out = self.classifier(review_graph)
        return out


class HGT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.model.graph_encoder.out_channels
        graph_encoder = get_graph_encoder('HGT')
        self.batch_size = cfg.train.batch_size

        self.graph_encoder = graph_encoder(cfg)
        self.classifier = nn.Linear(
            in_features=channels, out_features=cfg.model.num_classes)

    def forward(self, data):
        graph_dict = self.graph_encoder(data)

        review_graph = hetero_split(
            graph_dict, self.batch_size)
        out = self.classifier(review_graph)
        return out


class FULL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.model.graph_encoder.out_channels

        graph_encoder = get_graph_encoder('K-Genreformer')
        from models_.meta_encoder import MLP as meta_encoder
        self.graph_encoder = graph_encoder(cfg)
        self.meta_encoder = meta_encoder(cfg)

        self.activation = nn.LeakyReLU()

        self.movie_fusion = nn.Linear(
            in_features=channels*2, out_features=channels)
        self.movie_trm_fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=channels,
                                                                                 nhead=4,
                                                                                 dropout=0.3,
                                                                                 batch_first=True),
                                                      num_layers=2)

        self.review_fusion = nn.Linear(
            in_features=channels*2, out_features=channels)
        self.review_trm_fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=channels,
                                                                                  nhead=4,
                                                                                  dropout=0.3,
                                                                                  batch_first=True),
                                                       num_layers=2)

        self.user_semsantic_linear = nn.Linear(
            in_features=channels, out_features=channels)

        self.user_bias_fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=channels,
                                                                                 nhead=4,
                                                                                 dropout=0.3,
                                                                                 batch_first=True),
                                                      num_layers=2)
        self.user_final_fusion = nn.Linear(channels*3, channels)

        self.final_fusion = nn.Linear(channels*3, channels)
        if cfg.model.moe.type == 'new moe':
            self.moe = NewMoE(in_features=channels,
                              hidden_features=channels,
                              out_features=channels,
                              genre_num=cfg.genre_num+1,
                              device=cfg.device,
                              layers=cfg.model.moe.num_layers)
        elif cfg.model.moe.type == 'moe':
            self.moe = MoE(cfg=cfg)
        elif cfg.model.moe.type == 'soft_moe':
            self.moe = SoftMoE(in_features=channels,
                               out_features=channels,
                               num_experts=cfg.model.moe.num_experts,
                               slots_per_expert=2)
        elif cfg.model.moe.type == 'mlp':
            self.moe = nn.Linear(in_features=channels, out_features=channels)
        else:
            raise ValueError(f'Invalid MOE type {cfg.model.moe.type}')
        
        self.moe_type = cfg.model.moe.type

        self.fla = nn.Flatten()

        self.classifier = nn.Linear(
            in_features=channels, out_features=cfg.model.num_classes)

    def forward(self, data):
        meta_feature, edge_index, edge_type, genre, genre_mask = data.meta_feature, data.edge_index, data.edge_type, data.genre, data.genre_mask
        user_bias_semantic = data.user_bias_semantic

        graph_feature = data.x
        graph_feature = self.graph_encoder(
            graph_feature, edge_index, edge_type, genre, genre_mask)
        meta_feature = self.meta_encoder(meta_feature)

        # To split movie, user, review feature from the whole
        movie_graph, user_graph, review_graph = split(
            graph_feature, data.movie_map, data.user_map, data.review_map)
        movie_meta, user_meta, review_meta = split(
            meta_feature, data.movie_map, data.user_map, data.review_map)

        review = torch.stack((review_graph, review_meta), dim=1)
        # review = torch.cat((review_graph, review_meta), dim=1)
        review = self.activation(self.review_trm_fusion(review))
        review = self.fla(review)
        review = self.activation(self.review_fusion(review))

        movie = torch.stack((movie_graph, movie_meta), dim=1)
        # # movie = torch.cat((movie_graph, movie_meta), dim=1)
        movie = self.activation(self.movie_trm_fusion(movie))
        movie = self.fla(movie)
        movie = self.activation(self.movie_fusion(movie))

        user_bias_semantic = self.activation(
            self.user_semsantic_linear(user_bias_semantic))
        user_bias = torch.stack(
            (user_graph, user_bias_semantic, user_meta), dim=1)
        user_bias = self.activation(self.user_bias_fusion(user_bias))
        user_bias = self.fla(user_bias)
        user = self.activation(
            self.user_final_fusion(user_bias))

        # all = torch.stack((review, user, movie), dim=1)
        if self.moe_type == 'soft_moe':
            all = torch.stack((review, user, movie), dim=1)
            all = self.moe(all)
            all = self.activation(self.final_fusion(self.fla(all)))
            out = self.classifier(all)
            return out

        elif self.moe_type == 'new moe':
            all = torch.cat((review, user, movie), dim=1)
            all = self.activation(self.final_fusion(all))
            all = self.moe(all, data.final_genre)
            out = self.classifier(all)
            return out

        elif self.moe_type == 'moe':
            all = torch.cat((review, user, movie), dim=1)
            all = self.activation(self.final_fusion(all))
            all, loss = self.moe(all)
            out = self.classifier(all)
            return out, loss

        elif self.moe_type == 'mlp':
            all = torch.cat((review, user, movie), dim=1)
            all = self.activation(self.final_fusion(all))
            all = self.moe(all)
            out = self.classifier(all)
            return out
