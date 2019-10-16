import torch
import torch.nn as nn


def fc_layer(in_dim, out_dim, n_layers, compression_ratio,
             mid_activation=nn.ELU, out_activation=None, batch_norm=True):
    """
    in_dim - Dimensionality of input. 
    out_dim - Dimensionality of output. 
    n_layers - Number of layers. 
    compression_ratio - Factor to multiply # of units by for each layer. 
    mid_activation - Activation function to use in intermediate layers. 
    out_activation - Activation function (if any) to use in output layer. 
    batch_norm - Whether to use batch normalization or not. 
    """
    # Build sequential node.
    layers = []
    in_dim = in_dim

    # Add intermediate layers.
    for _ in range(n_layers - 1):

        # Layer output dimensionality determined by ratio.
        mid_dim = int(in_dim * compression_ratio)

        # Add linear layer with optional batch norm and activation.
        layers.append(nn.Linear(in_dim, mid_dim))

        if batch_norm:
            layers.append(nn.BatchNorm1d(mid_dim))

        if mid_activation is not None:
            layers.append(mid_activation())

        in_dim = mid_dim

    # Add output layer.
    layers.append(nn.Linear(mid_dim, out_dim))

    # Add output activation if desired.
    if out_activation is not None:
        layers.append(out_activation)

    return nn.Sequential(*layers)


def topk(scores, labels, k, float_type=torch.FloatTensor):
    """
    Method for computing top-k accuracy. 

    Returns avg top-k accuracy across batch 
    as well as vector containing 1's where
    predictions were correct and 0's otherwise. 
    """

    # First get top-k for each predictions.
    _, top_k = scores.topk(k, 1)

    # Repeat correct label so we can do comparison.
    labels = labels.view(-1, 1).repeat(1, k)

    # Compute which rows have a correct answer in top-k as well as avg.
    correct = torch.sum(top_k.eq(labels).type(float_type), 1)

    return correct


def init_params(model):

    params = []

    for p in model.parameters():
        if len(p.shape) >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.normal_(p)

        params.append(p)

    return params


class Squeeze(nn.Module):
    def forward(self, input):
        return input.squeeze()
