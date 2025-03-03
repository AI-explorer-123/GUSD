import torch
import torch.nn as nn


class NewMoE(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, genre_num, device, activation='LeakyReLU', layers=1):
        super(NewMoE, self).__init__()
        self.layers = layers
        self.device = device
        self.genre_num = genre_num
        self.activation = nn.LeakyReLU()

        # Initialize parameters for layers
        self.parameters_list = nn.ParameterList()
        for i in range(layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = out_features if i == layers - 1 else hidden_features

            # Parameters: [num_genres, in_dim, out_dim]
            params = nn.Parameter(torch.randn(
                genre_num, in_dim, out_dim, device=device))
            self.parameters_list.append(params)

    def forward(self, x, genres):
        batch_size, _ = x.size()

        for layer_idx in range(self.layers):
            params = self.parameters_list[layer_idx]
            in_dim, out_dim = params.size(1), params.size(2)

            # Expand input x to match the dimensions required for gathering weights [batch_size, num_genres, in_dim]
            # [batch_size, 3, in_dim]
            x_expanded = x.unsqueeze(1).expand(-1, 3, -1)

            # Expand genres_indices to have proper dimensions [batch_size, 3, in_dim, out_dim]
            genres_indices = genres.unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, in_dim, out_dim)
            # Expand params to have proper dimensions [batch_size, genre_num, in_dim, out_dim]
            params = params.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Gather the weights according to genres_indices
            expert_weights = torch.gather(params, 1, genres_indices)

            # Mask to exclude padding genres
            mask = (genres != 0).unsqueeze(-1)

            # Apply expert weights using einsum, respecting mask
            weighted_outputs = torch.einsum(
                'bij, bijk->bik', x_expanded, expert_weights) * mask.float()

            # Aggregate outputs, dividing by valid mask count to average properly
            new_x = weighted_outputs.sum(
                dim=1) / mask.sum(dim=1).float().clamp(min=1e-9)

            # Apply activation function
            x = self.activation(new_x)

        return x