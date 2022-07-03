from typing import List, Tuple
import torch.nn as nn
import torch
from collections import OrderedDict


class SingleInputEuclideanMLP(nn.Sequential):
    """
    Euclidean MLP with single input.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: List[int],
                 out_features: int,
                 bias: bool = True):

        layers = OrderedDict()

        hidden_features.append(out_features)
        prev = in_features
        for i, hf in enumerate(hidden_features):
            layers[f'HyperbolicLinear{i}'] = nn.Linear(prev, hf, bias)
            prev = hf

        super().__init__(layers)

    def forward(self, x: torch.Tensor):
        x = self.ffn(x)
        return x


class DoubleInputEuclideanMLP(nn.Module):
    """
    Euclidean MLP with double input (suited to distance prediction).
    """

    def __init__(self,
                 in_features: Tuple[int, int],
                 hidden_features: List[int],
                 out_features: int,
                 bias: bool = True):
        super().__init__()

        in_features = sum(in_features)
        self.ffn = SingleInputEuclideanMLP(in_features, hidden_features, out_features, bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = torch.cat([x1, x2], dim=1)
        x = self.ffn(x)
        return x
