import numpy as np
import IPython
from IPython import display
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import torch
import torch.nn as nn


class S_GC(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size):
        super(S_GC, self).__init__()
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * s_kernel_size,
            kernel_size=1,
        )

    def forward(self, x, A, att_edge=None):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous()


class S_GC_att_edge(nn.Module):
    def __init__(self, in_channels, out_channels, s_kernel_size, num_att_edge):
        super(S_GC_att_edge, self).__init__()
        self.num_att_edge = num_att_edge
        self.s_kernel_size = s_kernel_size + num_att_edge
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * self.s_kernel_size,
            kernel_size=1,
        )

    def forward(self, x, A, att_edge):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x1 = x[:, : self.s_kernel_size - self.num_att_edge, :, :, :]
        x2 = x[:, -self.num_att_edge :, :, :, :]
        x1 = torch.einsum("nkctv,kvw->nctw", (x1, A))
        x2 = torch.einsum("nkctv,nkvw->nctw", (x2, att_edge))
        x_sum = x1 + x2

        return x_sum


class STGC_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        s_kernel_size,
        t_kernel_size,
        dropout,
        A_size,
        num_att_edge=0,
        use_att_edge=False,
    ):
        super(STGC_Block, self).__init__()
        # 空間グラフ畳み込み attention edgeありかなしか
        if not use_att_edge:
            self.sgc = S_GC(
                in_channels=in_channels,
                out_channels=out_channels,
                s_kernel_size=s_kernel_size,
            )
        else:
            self.sgc = S_GC_att_edge(
                in_channels=in_channels,
                out_channels=out_channels,
                s_kernel_size=s_kernel_size,
                num_att_edge=num_att_edge,
            )

        # Learnable weight matrix M エッジに重みを与えます. どのエッジが重要かを学習します.
        self.M = nn.Parameter(torch.ones(A_size))

        # 時間グラフ畳み込み
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel_size, 1),
                (stride, 1),
                ((t_kernel_size - 1) // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # 残差処理
        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, A, att_edge):
        x = self.tgc(self.sgc(x, A * self.M, att_edge)) + self.residual(x)
        return x
