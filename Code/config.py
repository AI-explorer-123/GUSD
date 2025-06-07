from yacs.config import CfgNode as CN
import argparse
import os


def set_cfg(cfg):
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    cfg.device = 1
    cfg.seed = 0
    cfg.dataset = 'imdb'
    cfg.imdb_path = './Data/processed_imdb_data'
    cfg.lcs_path = './Data/processed_lcs_data'
    cfg.mask = 'all'
    cfg.feature = 'all'
    cfg.lm_model = 'roberta'
    cfg.flooding = False
    cfg.flooding_gate = 0.33
    cfg.genre_num = 30
    cfg.fine_tuning = True
    cfg.train = CN()
    cfg.model = CN()

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train.name = None
    cfg.train.save_dir = '/data3/whr/zhk/Spoiler_Detection/code/MOESD/model'
    cfg.train.epochs = 20
    cfg.train.lr = 1e-4
    cfg.train.weight_decay = 1e-5
    cfg.train.batch_size = 64
    cfg.train.num_hops = 2
    cfg.train.weight = 1.3
    cfg.train.kg_model = 'CompGCN'
    cfg.train.test_per_epoch = 4
    cfg.train.multi_test = False
    cfg.train.note = ''
    cfg.train.patience = 3
    cfg.train.warmup_epochs = 5

    # ------------------------------------------------------------------------ #
    # Basic model opuse_genretions
    # ------------------------------------------------------------------------ #
    cfg.model.name = 'hgt'
    cfg.model.num_classes = 2
    cfg.model.dropout = 0.3
    cfg.model.walk_length = 3
    cfg.model.k = 40
    cfg.model.pe_dim = 128
    cfg.model.review_movie_fusion_size = 2048
    cfg.model.user_bias_fusion_size = 2048
    cfg.model.kg_channels_1 = 64
    cfg.model.kg_channels_2 = 50
    cfg.model.activation = 'lrelu'
    cfg.model.use_two_moe = True
    cfg.model.use_one_moe = False
    cfg.model.use_kg = True
    cfg.model.trm_layers = 2
    # zst
    cfg.model.hop_adj = False
    cfg.model.hop_num = 2
    cfg.model.ratio = 0.1
    cfg.model.num_layers_per_hop = 2
    cfg.model.hop_decay = 0.3

    # ------------------------------------------------------------------------ #
    # Graph encoder options
    # ------------------------------------------------------------------------ #
    cfg.model.graph_encoder = CN()
    cfg.model.graph_encoder.name = 'K-Genreformer'
    cfg.model.graph_encoder.genreformer_gnn = 'GAT'
    cfg.model.graph_encoder.activation = 'lrelu'
    cfg.model.graph_encoder.layers = 2
    cfg.model.graph_encoder.in_channels = 1024
    cfg.model.graph_encoder.hidden_channels = 1024
    cfg.model.graph_encoder.out_channels = 1024
    cfg.model.graph_encoder.dropout = 0.3
    cfg.model.graph_encoder.num_rels = 5
    cfg.model.graph_encoder.heads = 4
    cfg.model.graph_encoder.pooling = 'mean'

    # ------------------------------------------------------------------------ #
    # Meta encoder options
    # ------------------------------------------------------------------------ #
    cfg.model.meta_encoder = CN()
    cfg.model.meta_encoder.name = 'MLP'
    cfg.model.meta_encoder.activation = 'lrelu'
    cfg.model.meta_encoder.layers = 3
    cfg.model.meta_encoder.in_channels = 6
    cfg.model.meta_encoder.hidden_channels = 512
    cfg.model.meta_encoder.out_channels = 1024


    # ------------------------------------------------------------------------ #
    # Final MoE options
    # ------------------------------------------------------------------------ #
    cfg.model.moe = CN()
    cfg.model.moe.type = 'new moe'
    cfg.model.moe.noisy_gating = True
    cfg.model.moe.num_experts = 32
    cfg.model.moe.slots_per_expert = 2
    cfg.model.moe.out_channels = 2048*3
    cfg.model.moe.in_channels = 2048*3
    cfg.model.moe.hidden_channels = 2048*2
    cfg.model.moe.k = 1
    cfg.model.moe.num_head = 4
    cfg.model.moe.name = 'MLP'
    cfg.model.moe.num_layers = 2
    cfg.model.moe.dropout = 0.3
    return cfg


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    if cfg.dataset == 'lcs':
        specific = ['model.meta_encoder.in_channels', 6,
                    'lm_model', 'roberta',
                    'model.graph_encoder.in_channels', 768,
                    'model.graph_encoder.hidden_channels', 768,
                    'model.graph_encoder.out_channels', 768,
                    'model.meta_encoder.out_channels', 768,
                    'genre_num', 30]
    elif cfg.dataset == 'imdb':
        specific = ['model.meta_encoder.in_channels', 3,
                    'fine_tuning', False,
                    'lm_model', 'bge-large',
                    'model.graph_encoder.in_channels', 1024,
                    'model.graph_encoder.hidden_channels', 1024,
                    'model.graph_encoder.out_channels', 1024,
                    'model.meta_encoder.out_channels', 1024,
                    'genre_num', 21]
        
    cfg.merge_from_list(specific)

    return cfg


cfg = set_cfg(CN())
# print(cfg)
