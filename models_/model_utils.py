import torch
import torch_geometric.nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected
from torch_sparse import SparseTensor


def to_edge_index_list(edge_index, edge_type, num_relations):
    edge_index_list = []
    for i in range(num_relations):
        edge_index_list.append(edge_index[:, edge_type == i])
    return edge_index_list


def transpose(X, r):
    X = X.reshape(X.shape[0], -1, r)
    X = X.permute(0, 2, 1)
    return X


def Transpose(X, h, r):
    X = X.reshape(X.shape[0], r, h, -1)
    X = X.permute(1, 2, 0, 3)
    return X


def split(feature, movie_map, user_map, review_map):
    movie_feature = feature[movie_map]
    user_feature = feature[user_map]
    review_feature = feature[review_map]
    return movie_feature, user_feature, review_feature


def hetero_split(feature_dict, batch_size):
    # movie_feature = feature_dict["movie"][movie_map]
    # user_feature = feature_dict["user"][user_map]
    review_feature = feature_dict["review"][:batch_size]
    return review_feature


def get_graph_encoder(model_name):
    if model_name == "GPS":
        from models_.graph_encoder.models import GPS as graph_encoder
    elif model_name == "SimpleHGN":
        from models_.graph_encoder.models import SimpleHGN as graph_encoder
    elif model_name == "HGT":
        from models_.graph_encoder.models import HGT as graph_encoder
    elif model_name == "GAT":
        from models_.graph_encoder.models import GAT as graph_encoder
    elif model_name == "RGAT":
        from models_.graph_encoder.models import RGAT as graph_encoder
    elif model_name == "GT":
        from models_.graph_encoder.models import GT as graph_encoder
    elif model_name == "hetero_genreformer":
        from models_.graph_encoder.models import hetero_genreformer as graph_encoder
    elif model_name == "genreformer":
        from models_.graph_encoder.models import genreformer as graph_encoder
    elif model_name == "K-Genreformer":
        from models_.graph_encoder.models import K_Genreformer as graph_encoder
    elif model_name == 'K-GAT':
        from models_.graph_encoder.models import GAT_KH as graph_encoder
    return graph_encoder


def get_graph_conv(conv_name):
    if conv_name == "SimpleHGN":
        from models_.graph_encoder.models import SimpleHGN_conv as graph_conv
    if conv_name == "GAT":
        from torch_geometric.nn import GATConv as graph_conv
    return graph_conv


def genre_forward(x, genre_num, genres):
    '''
    Turn into Sequances
    '''
    # batch_size, num_features = x.shape
    # genre_counts = [(genres == genre_id + 1).sum().item()
    #                 for genre_id in range(genre_num)]
    # max_comments_per_genre = np.array(genre_counts).max()

    # all_genre = torch.zeros(
    #     (genre_num, max_comments_per_genre, num_features), device=x.device)
    # mask = torch.ones((genre_num, max_comments_per_genre),
    #                   dtype=torch.bool, device=x.device)

    # genre_positions = torch.zeros(
    #     genre_num, dtype=torch.long, device=x.device)
    # for idx, sample_genres in enumerate(genres):
    #     for genre_id in sample_genres:
    #         if genre_id > 0:
    #             genre_id = int(genre_id.item())
    #             pos = genre_positions[genre_id - 1]
    #             all_genre[genre_id - 1, pos, :] = x[idx]
    #             mask[genre_id - 1, pos] = False
    #             genre_positions[genre_id - 1] += 1
    # mask = torch.cat(
    #     (torch.zeros((genre_num, 1), dtype=torch.bool, device=x.device), mask), dim=1)

    # return all_genre, mask

    ''''
    Simply Add
    '''
    batch_size, num_features = x.size()
    genre_feature = torch.zeros(
        genre_num, num_features, device=x.device, dtype=x.dtype)

    # 调整 genres 中的体裁索引以从 0 开始
    genres = genres - 1

    # 创建一个扩展的 x tensor，每行对应于 genres 中的每个体裁
    expanded_x = x.unsqueeze(1).expand(-1, 3, -1)

    # 使用 mask 来排除 padding 体裁（即索引为 -1 的体裁）
    valid_mask = genres >= 0  # 将无效体裁设置为 0，后续使用 valid_mask 过滤
    valid_genres = (genres * valid_mask).view(-1, 1).expand(-1, num_features) # 变成[batch_size*3, num_features]
    valid_x = (expanded_x * valid_mask.unsqueeze(2)).reshape(-1, num_features) # 变成[batch_size*3*num_features, num_features]

    genre_feature.scatter_add_(0, valid_genres, valid_x)

    mask = genre_feature.sum(dim=1) == 0

    return genre_feature.unsqueeze(0), mask.unsqueeze(0)


def genre_backward(genres, genre_feature, pooling='mean'):
    genres = genres.long()

    zero_mask = (genres != 0).float().unsqueeze(-1)
    genre_feature = F.pad(genre_feature, pad=(0, 0,
                                              1, 0))
    expanded_genre_features = genre_feature[genres]
    expanded_genre_features = expanded_genre_features * zero_mask

    if pooling == 'mean':
        genre_embedding = expanded_genre_features.sum(dim=1) / \
            (genres != 0).sum(dim=1, keepdim=True)
    elif pooling == 'sum':
        genre_embedding = expanded_genre_features.sum(dim=1)
    elif pooling == 'max':
        genre_embedding = expanded_genre_features.max(dim=1)[0]

    return genre_embedding.to(genre_feature.device)


def k_hop_adj(edge_index, num_nodes, k, device, ratio=1):
    edge_index_ = to_undirected(edge_index, num_nodes=num_nodes)

    # 创建稀疏邻接矩阵
    adj = SparseTensor(row=edge_index_[0], col=edge_index_[
                       1], sparse_sizes=(num_nodes, num_nodes))

    adj_n = adj
    edge_index_k_hops = []
    edge_index_k_hops.append(edge_index)
    for j in range(2, k+1):

        adj_n = adj_n @ adj

        # 转换edge_index格式
        row, col, _ = adj_n.coo()
        # 返回时，>1即为出现一次，值结果并不影响出现的边数目，因为coo返回的值被接受为 _ 存储的是对应点的值
        edge_index_k_hop = torch.stack([row, col], dim=0)

        num_edges = edge_index_k_hop.size(1)
        num_samples = int(num_edges * ratio)

        # 随机生成要保留的边的索引
        indices = torch.randperm(num_edges, device=device)[:num_samples]

        # 使用随机选择的索引获取边
        edge_index_k_hop = edge_index_k_hop[:, indices]
        edge_index_k_hops.append(edge_index_k_hop)
    return edge_index_k_hops


def k_hop_neighbors_with_min_distance(edge_index, num_nodes, k, device, ratio=1):
    # 创建邻接矩阵的稀疏表示
    max_k = 100

    edge_index_ = to_undirected(edge_index, num_nodes=num_nodes)  # 无向图

    values = torch.ones(edge_index_.size(1), device=device)
    adjacency_matrix = torch.sparse.FloatTensor(
        edge_index_, values, torch.Size([num_nodes, num_nodes]))

    # 初始化距离矩阵D
    D = torch.full((num_nodes, num_nodes), float('inf'), device=device)
    D.scatter_(1, torch.arange(
        num_nodes, device=device).view(1, -1), 0)  # 设置自环距离为0\

    edge_index_k_hops = []
    edge_index_k_hops.append(edge_index)
    # 进行k次幂运算，更新距离矩阵D
    A_power = adjacency_matrix.clone()
    for i in range(1, k+1):
        # 对于直接相连的节点，距离设置为1
        if i == 1:
            D[A_power.coalesce().indices()[0], A_power.coalesce().indices()[1]] = 1
        else:
            A_power = torch.sparse.mm(A_power, adjacency_matrix)  # A^i
            # 将所有大于0的值置为1，其余保持不变
            A_binary = (A_power.to_dense() > 0).float()
            # 将非零值乘以i，0值设置为inf
            A_distance = A_binary * i + (1 - A_binary) * max_k  # 避免出现0*inf=nan

            # A_distance = torch.where(A_binary > 0, A_binary * i, torch.tensor(float('inf')))

            # 更新D矩阵，只保留最小值
            D = torch.min(D, A_distance)
            edge_index_k_hop = D == torch.full(
                (num_nodes, num_nodes), i, device=device)
            edge_index_k_hop = edge_index_k_hop.nonzero().t().contiguous()

            num_edges = edge_index_k_hop.size(1)
            num_samples = int(num_edges * ratio)
            indices = torch.randperm(num_edges, device=device)[:num_samples]
            edge_index_k_hop = edge_index_k_hop[:, indices]
            edge_index_k_hops.append(edge_index_k_hop)

    return edge_index_k_hops
