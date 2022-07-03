from typing import Union, Tuple, List
import torch
import torch.nn as nn
from collections import OrderedDict
from geoopt import ManifoldParameter, ManifoldTensor, PoincareBallExact, PoincareBall


class PoincareLinear(nn.Module):
    def __init__(self,
                 ball: Union[PoincareBallExact, PoincareBall],
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.ball = ball
        self.bias = ManifoldParameter(torch.empty((1, out_features)), manifold=self.ball) if bias else None

    def forward(self, x: ManifoldTensor):
        x = self.ball.logmap0(x)
        x = self.fc(x)
        x = self.ball.expmap0(x)

        if self.bias is not None:
            x = self.ball.mobius_add(x, self.bias)

        return x

    @torch.no_grad()
    def reset_parameters(self):
        direction = torch.randn_like(self.bias)
        direction /= direction.norm(dim=-1, keepdim=True)
        distance = torch.empty_like(self.bias[..., 0]).normal_()
        self.bias.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))


class PoincareConcat(nn.Module):
    def __init__(self,
                 ball: Union[PoincareBallExact, PoincareBall],
                 in_features: Tuple[int, int],
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        self.ball = ball
        in1, in2 = in_features
        self.mfc1 = PoincareLinear(ball, in1, out_features, False)
        self.mfc2 = PoincareLinear(ball, in2, out_features, False)
        self.bias = ManifoldParameter(torch.empty((1, out_features)), manifold=self.ball) if bias else None

    def forward(self, x1: ManifoldTensor, x2: ManifoldTensor):
        x1 = self.mfc1(x1)
        x2 = self.mfc2(x2)
        x = self.ball.mobius_add(x1, x2)

        if self.bias is not None:
            x = self.ball.mobius_add(x, self.bias)

        return x

    @torch.no_grad()
    def reset_parameters(self):
        direction = torch.randn_like(self.bias)
        direction /= direction.norm(dim=-1, keepdim=True)
        distance = torch.empty_like(self.bias[..., 0]).normal_()
        self.bias.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))


# class MobiusReLU(nn.Module):
#     def __init__(self, curv):
#         self.curv = curv
#         super().__init__()
#
#     def forward(self, x):
#         x = mobius(F.relu, self.curv)(x)
#         return x


# class HyperbolicLinear(nn.Module):
#     """
#     Wrapper for HyperbolicLinear with activation.
#     """
#     def __init__(self, input_dim, output_dim, activation, bias, curv):
#         super().__init__()
#         self.fc = MobiusLinear(input_dim, output_dim, bias, curv=curv)
#         if activation == 'relu':
#             self.activation = MobiusReLU(curv)
#         elif activation == 'None':
#             self.activation = lambda x: x
#         else:
#             raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')
#
#     def forward(self, x):
#         return self.activation(self.fc(x))
#
#
# class HyperbolicLinearSkip(nn.Module):
#     """
#     Wrapper for HyperbolicLinear with skip connection after activation.
#     """
#     def __init__(self, input_dim, output_dim, activation, bias, curv):
#         super().__init__()
#         self.fc = MobiusLinear(input_dim, output_dim, bias, curv=curv)
#         if activation == 'relu':
#             self.activation = MobiusReLU(curv)
#         elif activation == 'None':
#             self.activation = lambda x: x
#         else:
#             raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')
#
#     def forward(self, x):
#         y = self.fc(x)
#         y = self.activation(y)
#         return y + x


class SingleInputPoincareMLP(nn.Sequential):
    """
    Hyperbolic feed-forward network with single input.
    """

    def __init__(self,
                 ball: Union[PoincareBallExact, PoincareBall],
                 in_features: int,
                 hidden_features: List[int],
                 out_features: int,
                 bias: bool = True):

        self.ball = ball

        layers = OrderedDict()

        hidden_features.append(out_features)
        prev = in_features
        for i, hf in enumerate(hidden_features):
            layers[f'HyperbolicLinear{i}'] = PoincareLinear(ball, prev, hf, bias)
            prev = hf

        super().__init__(layers)

    def forward(self, x: ManifoldTensor):
        x = self.ffn(x)
        return x


class DoubleInputPoincareMLP(nn.Module):
    """
    Hyperbolic feed-forward network with double input (suited to distance prediction).
    """
    def __init__(self,
                 ball: Union[PoincareBallExact, PoincareBall],
                 in_features: Tuple[int, int],
                 hidden_features: List[int],
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        self.ball = ball

        if hidden_features:
            ffn_in_features, ffn_hidden_features = hidden_features[0], hidden_features[1:]
            self.concat_layer = PoincareConcat(ball, in_features, ffn_in_features, bias)
            self.ffn = SingleInputPoincareMLP(ball, ffn_in_features, ffn_hidden_features, out_features, bias)
        else:
            self.concat_layer = PoincareConcat(ball, in_features, out_features, bias)
            self.ffn = lambda x: x

    def forward(self, x1: ManifoldTensor, x2: ManifoldTensor):
        x = self.concat_layer(x1, x2)
        x = self.ffn(x)
        return x
