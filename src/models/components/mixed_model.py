from typing import Tuple, List, Union
import torch.nn as nn
import torch
from euclidean_mlp import SingleInputEuclideanMLP
from hyperbolic_mlp import DoubleInputPoincareMLP
from geoopt.manifolds.stereographic import PoincareBall, PoincareBallExact


class TrueEmbedding(nn.Module):
    def __init__(self,
                 ball: Union[PoincareBall, PoincareBallExact]):
        super().__init__()
        self.ball = ball

    def forward(self, x1, x2):
        return self.ball.mobius_add(-x1, x2)


class TrueHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2 * torch.arctanh(torch.linalg.norm(x, dim=1))


class MixedModel(nn.Module):
    def __init__(self,
                 hnn: Union[DoubleInputPoincareMLP, TrueEmbedding],
                 enn: Union[SingleInputEuclideanMLP, TrueHead]):
        super().__init__()
        self.hnn = hnn
        self.enn = enn

    def forward(self, x1, x2):
        x = self.hnn(x1, x2)
        x = self.hnn.ball.logmap0(x)
        x = self.enn(x)
        return x


class StandardMixedModel(MixedModel):
    def __init__(self,
                 hnn_ball: Union[PoincareBall, PoincareBallExact],
                 in_features: Tuple[int, int],
                 hnn_hidden_features: List[int],
                 inter_features: int,
                 enn_hidden_features: List[int],
                 out_features: int):
        assert in_features[0] == in_features[1]
        hnn = DoubleInputPoincareMLP(hnn_ball, in_features, hnn_hidden_features, inter_features, True)
        enn = SingleInputEuclideanMLP(inter_features, enn_hidden_features, out_features, True)
        super().__init__(hnn, enn)


class TrueModel(MixedModel):
    def __init__(self,
                 hnn_ball: Union[PoincareBall, PoincareBallExact]):
        hnn = TrueEmbedding(hnn_ball)
        enn = TrueHead()
        super().__init__(hnn, enn)


class TrueEmbeddingModel(MixedModel):
    def __init__(self,
                 hnn_ball: Union[PoincareBall, PoincareBallExact],
                 in_features: Tuple[int, int],
                 enn_hidden_features: List[int],
                 out_features: int):
        assert in_features[0] == in_features[1]
        hnn = TrueEmbedding(hnn_ball)
        enn = SingleInputEuclideanMLP(in_features[0], enn_hidden_features, out_features, True)
        super().__init__(hnn, enn)


class TrueHeadModel(MixedModel):
    def __init__(self,
                 hnn_ball: Union[PoincareBall, PoincareBallExact],
                 in_features: Tuple[int, int],
                 hnn_hidden_features: List[int],
                 inter_features: int):
        assert in_features[0] == in_features[1]
        hnn = DoubleInputPoincareMLP(hnn_ball, in_features, hnn_hidden_features, inter_features, True)
        enn = TrueHead()
        super().__init__(hnn, enn)
