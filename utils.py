import logging
import torch
import json
import numpy as np
import random
import os
import datetime
import time
import pytz
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models_.main_model import *

# ------------------------------------Loss Related---------------------------------------


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.alpha = self.alpha.to('cuda:3')
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Focal_Loss():
    def __init__(self, alpha=0.25, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels:标签
        """
        eps = 1e-7
        loss_1 = -1*self.alpha * \
            torch.pow((1-preds), self.gamma)*torch.log(preds+eps)*labels
        loss_0 = -1*(1-self.alpha)*torch.pow(preds, self.gamma) * \
            torch.log(1-preds+eps)*(1-labels)
        loss = loss_0+loss_1
        return torch.mean(loss)


# ------------------------------------Data Related---------------------------------------

def set_seed(seed: int):
    """ Set random seeds for numpy and torch"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def load_data(cfg):
    if cfg.dataset == 'lcs':
        data = load_lcs_data(cfg)
    elif cfg.dataset == 'imdb':
        data = load_imdb_data(cfg)
    else:
        raise NameError(f'There is no {cfg.dataset} dataset!')
    return data


def load_lcs_data(cfg):
    path = "/data3/whr/zhk/Spoiler_Detection/Data/processed_lcs_data/"

    base_edge_index = torch.load(
        path+"edge/base_edge_index.pt").to(torch.long)
    base_edge_type = torch.load(
        path+"edge/base_edge_type.pt").to(torch.long).squeeze()
    review_edge_index = torch.load(
        path+'edge/review_edge_index.pt').to(torch.long)
    review_edge_type = torch.load(
        path+'edge/review_edge_type.pt').to(torch.long).squeeze()
    similar_movie_edge_index = torch.load(
        path+'edge/similar_movie_edge_index.pt').to(torch.long)
    similar_movie_edge_type = torch.load(
        path+'edge/similar_movie_edge_type.pt').to(torch.long).squeeze()

    edge_index = torch.cat(
        (base_edge_index, similar_movie_edge_index), dim=1)
    edge_type = torch.cat(
        (base_edge_type, similar_movie_edge_type))

    # semantic_feature_1 = torch.load(
    #     path+f'final_semantic/{cfg.lm_model}/all_semantic.pt', map_location='cpu')
    # semantic_feature_2 = torch.load(path + 'final_semantic/dyg_embedding.pt', map_location='cpu')
    # semantic_feature = torch.cat(
    #     (semantic_feature_1, semantic_feature_2), dim=1)
    if cfg.fine_tuning == True:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/ft_semantic.pt', map_location='cpu')
    else:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/all_semantic.pt', map_location='cpu')
    # user_semantic = torch.load(path + 'final_semantic/dyg_embedding.pt')
    # semantic_feature = torch.cat(
    #     (semantic_feature[:147191], user_semantic[147191:406896], semantic_feature[406896:]), dim=0)

    meta_feature = torch.load(path +
                              'meta.pt', map_location='cpu')

    label = torch.load(path +
                       "label.pt").to(torch.int64)

    train_mask = torch.load(path + "mask/train_index.pt").to(torch.long)
    val_mask = torch.load(path + "mask/val_index.pt").to(torch.long)
    test_mask = torch.load(path + "mask/test_index.pt").to(torch.long)

    review_movie_map = torch.load(path +
                                  "map/review_movie_map.pt")
    review_user_map = torch.load(path +
                                 'map/review_user_map.pt')

    # user_bias_semantic = torch.load(path+
    #     'final_semantic/user_bias_semantic.pt', map_location='cpu')

    user_bias_semantic = torch.load(path +
                                    'final_semantic/user_bias.pt', map_location='cpu')  # user_bias_semantic = torch.load(path +
    # 'final_semantic/dyg_time_user_bias.pt', map_location='cpu')

    review_genre = torch.load(path + 'review_genre.pt')
    movie_genre = torch.load(path + 'movie_genre.pt')
    genre = torch.cat((review_genre, movie_genre), dim=0)
    data = Data(x=semantic_feature,
                meta_feature=meta_feature,
                y=label,
                edge_index=edge_index,
                edge_type=edge_type,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                review_movie_map=review_movie_map,
                review_user_map=review_user_map,
                user_bias_semantic=user_bias_semantic,
                genre=genre,
                user_start=147191,
                review_start=406896,
                review_num=1860715).to('cpu')
    return data


def load_imdb_data(cfg):
    path = "/data3/whr/zhk/Spoiler_Detection/Data/processed_imdb_data/"

    edge_index = torch.load(path+"edge/base_edge_index.pt").to(torch.long)
    edge_type = torch.load(
        path+"edge/base_edge_type.pt").to(torch.long).squeeze()

    if cfg.fine_tuning == True:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/ft_semantic.pt', map_location='cpu')
    else:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/all_semantic.pt', map_location='cpu')
    # user_semantic = torch.load(path + 'final_semantic/dyg_embedding.pt')
    # semantic_feature = torch.cat(
    #     (semantic_feature[:147191], user_semantic[147191:406896], semantic_feature[406896:]), dim=0)

    meta_feature = torch.load(path + 'meta.pt', map_location='cpu')

    label = torch.load(path + "label.pt").to(torch.int64)

    train_mask = torch.load(path + "mask/train_index.pt").to(torch.long)
    val_mask = torch.load(path + "mask/val_index.pt").to(torch.long)
    test_mask = torch.load(path + "mask/test_index.pt").to(torch.long)

    review_movie_map = torch.load(path + "map/review_movie_map.pt")
    review_user_map = torch.load(path + 'map/review_user_map.pt')

    user_bias_semantic = torch.load(path +
                                    'final_semantic/user_bias.pt', map_location='cpu')

    review_genre = torch.load(path + 'review_genre.pt')
    movie_genre = torch.load(path + 'movie_genre.pt')
    genre = torch.cat((review_genre, movie_genre), dim=0)
    data = Data(x=semantic_feature,
                meta_feature=meta_feature,
                y=label,
                edge_index=edge_index,
                edge_type=edge_type,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                review_movie_map=review_movie_map,
                review_user_map=review_user_map,
                genre=genre,
                user_bias_semantic=user_bias_semantic,
                user_start=1572,
                review_start=264976,
                review_num=573906).to('cpu')
    return data


def get_map(all_nodes, review_ids, review_movie_map, review_user_map):
    review_ids = torch.LongTensor(review_ids)
    movies_id = review_movie_map[review_ids].numpy().tolist()
    users_id = review_user_map[review_ids].numpy().tolist()

    movie_map = [torch.where(all_nodes == movie_id) for movie_id in movies_id]
    movie_map = torch.Tensor(movie_map).to(int).squeeze()
    user_map = [torch.where(all_nodes == user_id) for user_id in users_id]
    user_map = torch.Tensor(user_map).to(int).squeeze()
    return movie_map, user_map, users_id, movies_id


def get_batch(data, num_hops, sub_nodes):
    nodes_idx, sub_edge_index, review_map, edge_mask = k_hop_subgraph(
        sub_nodes, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True)
    sub_x = data.x[nodes_idx]
    sub_meta = data.meta_feature[nodes_idx]
    sub_y = data.y[sub_nodes]
    sub_edge_type = data.edge_type[edge_mask]

    movie_map, user_map, user_id, movie_id = get_map(all_nodes=nodes_idx,
                                                     review_ids=sub_nodes,
                                                     review_movie_map=data.review_movie_map,
                                                     review_user_map=data.review_user_map)
    batch_user_bias_semantic = data.user_bias_semantic[torch.Tensor(
        user_id).to(int)-147191]
    # batch_kg = data.kg[torch.Tensor(movie_id).to(int)]

    batch = Data(x=sub_x,
                 y=sub_y,
                 meta_feature=sub_meta,
                 edge_index=sub_edge_index,
                 edge_type=sub_edge_type,
                 review_map=review_map,
                 movie_map=movie_map,
                 user_map=user_map)
    return batch


def get_genre_batch(data, num_hops, sub_nodes):
    sub_nodes = torch.LongTensor(sub_nodes)
    nodes_idx, sub_edge_index, review_map, edge_mask = k_hop_subgraph(
        sub_nodes, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True)
    sub_x = data.x[nodes_idx]
    sub_meta = data.meta_feature[nodes_idx]
    sub_y = data.y[sub_nodes]
    sub_edge_type = data.edge_type[edge_mask]

    review_genre_mask = torch.where(
        nodes_idx >= data.review_start, True, False)
    movie_genre_mask = torch.where(nodes_idx < data.user_start, True, False)
    genre_mask = review_genre_mask + movie_genre_mask
    batch_genre = torch.cat(
        (data.genre[nodes_idx[review_genre_mask]-data.review_start], data.genre[nodes_idx[movie_genre_mask]+data.review_num]), dim=0)
    batch_review_genre = data.genre[torch.LongTensor(
        sub_nodes)-data.review_start]

    movie_map, user_map, user_id, movie_id = get_map(all_nodes=nodes_idx,
                                                     review_ids=[
                                                         sub_node for sub_node in sub_nodes],
                                                     review_movie_map=data.review_movie_map,
                                                     review_user_map=data.review_user_map)
    batch_user_bias_semantic = data.user_bias_semantic[torch.Tensor(
        user_id).to(int)-data.user_start]

    batch = Data(x=sub_x,
                 y=sub_y,
                 meta_feature=sub_meta,
                 edge_index=sub_edge_index,
                 edge_type=sub_edge_type,
                 review_map=review_map,
                 movie_map=movie_map,
                 user_map=user_map,
                 genre=batch_genre,
                 genre_mask=genre_mask,
                 final_genre=batch_review_genre,
                 user_bias_semantic=batch_user_bias_semantic)
    return batch
# ------------------------------------Str to Func---------------------------------------


def get_model(model_name, cfg):
    if model_name == 'full':
        model = FULL(cfg=cfg)
    elif model_name == 'mlp':
        model = MLP(cfg=cfg)
    elif model_name == 'gnn':
        model = GNN(cfg=cfg)
    elif model_name == 'hgt':
        model = HGT(cfg=cfg)
    return model


# ------------------------------------Time Related---------------------------------------

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


# ------------------------------------Hetero Related---------------------------------------

def load_hetero_data(cfg):
    mask = cfg.mask
    path = f"/data3/whr/zhk/Spoiler_Detection/Data/processed_{cfg.dataset}_data/"
    edge_index = torch.load(path+"edge/base_edge_index.pt").to(torch.long)
    edge_type = torch.load(
        path+"edge/base_edge_type.pt").to(torch.long).squeeze()

    if cfg.fine_tuning == True:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/ft_semantic.pt', map_location='cpu')
    else:
        semantic_feature = torch.load(
            path+f'final_semantic/{cfg.lm_model}/all_semantic.pt', map_location='cpu')

    label = torch.load(path +
                       "label.pt").to(torch.int64)
    train_mask = torch.load(path + "mask/train_index.pt").to(torch.long)
    val_mask = torch.load(path + "mask/val_index.pt").to(torch.long)
    test_mask = torch.load(path + "mask/test_index.pt").to(torch.long)

    review_movie_map = torch.load(path +
                                  "map/review_movie_map.pt")
    review_user_map = torch.load(path +
                                 'map/review_user_map.pt')-1572

    data = HeteroData()
    data['user'].x = semantic_feature[147191:406896]
    
    data['movie'].x = semantic_feature[:147191]

    data['review'].x = semantic_feature[406896:]
    data['review'].y = label[406896:]

    edge_index0 = edge_index[:, edge_type == 0]
    edge_index0[1] -= 406896
    edge_index1 = edge_index[:, edge_type == 3]
    edge_index1[0] -= 147191
    edge_index1[1] -= 406896
    edge_index2 = edge_index[:, edge_type == 1]
    edge_index2[0] -= 406896
    edge_index2[1] -= 147191

    data['movie', 'commented_by', 'review'].edge_index = edge_index0
    data['user', 'post', 'review'].edge_index = edge_index1
    data['review', 'posted_by', 'user'].edge_index = edge_index2

    data['review_movie_map'] = review_movie_map
    data['review_user_map'] = review_user_map

    data['train_mask'] = train_mask-406896
    data['val_mask'] = val_mask-406896
    data['test_mask'] = test_mask-406896
    return data


def load_hetero_map(data):
    review_ids = data['review'].n_id[:data['review'].batch_size]
    movies_id = data.review_movie_map[review_ids].numpy().tolist()
    users_id = data.review_user_map[review_ids].numpy().tolist()

    movie_map = [torch.where(data['movie'].n_id == movie_id)
                 for movie_id in movies_id]
    movie_map = torch.Tensor(movie_map).to(int).squeeze()
    user_map = [torch.where(data['user'].n_id == user_id)
                for user_id in users_id]
    user_map = torch.Tensor(user_map).to(int).squeeze()
    data['movie_map'] = movie_map
    data['user_map'] = user_map
    return data


# ------------------------------------Others---------------------------------------

class MetricsHandler(object):
    def __init__(self, metrics, patience: int, checkpoint_dir: str, train_name: str, seed: int, warmup_epochs=5):
        self.patience = patience
        self.counter = 0
        self.best_metrics, self.higher_betters = self.init_metrics(metrics)
        self.early_stop = False
        self.warmup_epochs = warmup_epochs

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f'Created checkpoint directory: {checkpoint_dir}')
        self.save_path = os.path.join(
            checkpoint_dir, f'{train_name}_seed{seed}.pth')

    def init_metrics(self, metrics: list):
        higher_betters = {}
        best_metrics = {}
        for metric_tuple in metrics:
            metric, higher_better = metric_tuple[0], metric_tuple[1]
            higher_betters[metric] = higher_better
            best_metrics[metric] = 0 if higher_better else 1
        return best_metrics, higher_betters

    def handle_metrics(self, metrics: list, model: nn.Module, epoch: int, test_loss: float = None, step: str = 'test'):
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[
                0], metric_tuple[1], self.higher_betters[metric_tuple[
                    0]]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                    self.best_metrics[metric_name] = metric_value
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                    self.best_metrics[metric_name] = metric_value
                else:
                    metrics_compare_results.append(False)

        if step == 'valid':
            print(f"         {step} loss: {test_loss:.4f}, " +
                  ", ".join([f"{name}: {value:.4f}" for name, value in metrics]))
        else:
            print(f"         {step} loss: {test_loss:.4f}, " + ", ".join([f"{name}: {value:.4f}" for name, value in metrics]) +
                  f", best_f1: {self.best_metrics.get('f1', 0):.4f}, best_auc: {self.best_metrics.get('auc', 0):.4f}, best_acc: {self.best_metrics.get('acc', 0):.4f}")

        if epoch >= self.warmup_epochs:
            self.earlystop_step(
                metrics_compare_results, model)

        return self.early_stop

    def earlystop_step(self, metrics_compare_results: list, model: nn.Module):
        if torch.any(torch.tensor(metrics_compare_results)):
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module):
        torch.save(model.state_dict(), self.save_path)
