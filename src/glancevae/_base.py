import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable

from torch.distributions import Normal


class FCLayers(nn.Module):
    """
    一个简化版的 FCLayers：
    - 每一层按照：Linear → BatchNorm? → LayerNorm? → Activation? → Dropout?
    - layers_dim 定义网络结构，例如 [in_dim, 128, 64]
    - inject_into_layer(i) 决定是否在第 i 层输入 concatenated covariates
    """

    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=128,
        n_layers=1,
        n_cov=0,  # covariate dimension
        inject_into_layer=False,
        activation_fn=nn.ReLU,
        dropout_rate=0.0,
        use_batch_norm=False,
        use_layer_norm=False,
        bias=True,
    ):
        super().__init__()

        self.n_cov = n_cov
        self.inject_into_layer = inject_into_layer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        layer_dict = collections.OrderedDict()

        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):

            # 如果需要注入 covariates，则增加输入维度
            if i == 0 or inject_into_layer:
                n_in = n_in + n_cov

            modules = []

            # Linear
            modules.append(nn.Linear(n_in, n_out, bias=bias))

            # BatchNorm
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001))

            # LayerNorm
            if use_layer_norm:
                modules.append(nn.LayerNorm(n_out, elementwise_affine=False))

            # Activation
            if activation_fn is not None:
                modules.append(activation_fn())

            # Dropout
            if dropout_rate > 0:
                modules.append(nn.Dropout(dropout_rate))

            # 组装一层
            layer_dict[f"Layer_{i}"] = nn.Sequential(*modules)

        self.fc_layers = nn.Sequential(layer_dict)

    def forward(self, x, cov=None):
        """
        x: shape (N, features)
        cov: shape (N, n_cov)，可选
        """
        for i, layer in enumerate(self.fc_layers):
            if (i == 0 or self.inject_into_layer) and cov is not None:
                x = torch.cat([x, cov], dim=1)
            x = layer(x)

        return x


class GaussianEncoder(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=128,
        n_layers=1,
        n_cov=0,  # covariate dimension
        inject_into_layer=False,
        activation_fn=nn.ReLU,
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        **kwargs,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_cov=n_cov,  # covariate dimension
            inject_into_layer=inject_into_layer,
            activation_fn=activation_fn,
            **kwargs,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_out)
        self.var_encoder = nn.Linear(n_hidden, n_out)

        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x, cov=None):

        # Parameters for latent distribution
        q = self.encoder(x, cov)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps

        qz = Normal(q_m, q_v.sqrt())
        z = qz.rsample()

        return q_m, q_v, z


class StudentTDecoder(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=128,
        n_layers=1,
        n_cov=0,  # covariate dimension
        inject_into_layer=True,
        var_eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__()

        self.var_eps = var_eps

        self.decoder = FCLayers(
            n_in=n_in,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_cov=n_cov,
            inject_into_layer=inject_into_layer,
            **kwargs,
        )

        # 2. 输出头：预测 Location (均值) 和 Scale (尺度)
        self.loc_decoder = nn.Linear(n_hidden, n_out)
        self.scale_decoder = nn.Linear(n_hidden, n_out)

        # 3. 核心改进：Student's t 的自由度 (Degrees of Freedom)
        # 这是一个 Learnable Parameter，形状为 (n_out,)，即每个 Gene 都有自己专属的 df
        # 初始化为 10.0 是经验值：既允许长尾（比高斯宽），又不至于方差无穷大（df < 2时）
        self.df = nn.Parameter(torch.ones(n_out) * 10.0)

    def forward(
        self,
        z: torch.Tensor,
        cov=None,
    ):
        # 1. 解码 hidden representation
        p = self.decoder(z, cov)

        # 2. 预测 Location (类似 Gaussian 的 mu)
        p_loc = self.loc_decoder(p)

        # 3. 预测 Scale (类似 Gaussian 的 sigma)
        # 使用 exp 保证正数，clamp_min 防止数值不稳定
        p_scale = torch.exp(self.scale_decoder(p)).clamp_min(self.var_eps)

        # 4. 获取自由度 df
        # 使用 softplus 确保 df 始终 > 0
        p_df = F.softplus(self.df)

        return p_loc, p_scale, p_df
