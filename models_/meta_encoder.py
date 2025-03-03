import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = cfg.model.meta_encoder.layers
        in_channels = cfg.model.meta_encoder.in_channels
        hidden_channels = cfg.model.meta_encoder.hidden_channels
        out_channels = cfg.model.meta_encoder.out_channels
        
        if cfg.model.meta_encoder.activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        
        if self.layers == 1:
            self.linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        else:
            self.linear1 = nn.Linear(in_features=in_channels, out_features=hidden_channels)
            middle_layers = nn.Sequential(nn.Linear(in_features=hidden_channels, out_features=hidden_channels), self.activation)
            self.mlps = nn.Sequential(*[middle_layers for _ in range(self.layers-2)])
            self.linear2 = nn.Linear(in_features=hidden_channels, out_features=out_channels)
        
    def forward(self, x):
        if self.layers == 1:
            x = self.activation(self.linear(x))
        else:
            x = self.activation(self.linear1(x))
            try:
                x = self.mlps(x)
            except:
                pass
            x = self.activation(self.linear2(x))
        return x
        