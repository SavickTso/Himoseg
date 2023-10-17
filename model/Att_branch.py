import torch.nn as nn
import torch.nn.functional as F
from model import STGCN


class AttentionBranch(nn.Module):
    def __init__(
        self,
        config,
        num_classes,
        num_att_edge,
        s_kernel_size,
        t_kernel_size,
        dropout,
        A_size,
    ):
        super(AttentionBranch, self).__init__()
        # STGC-Block config
        kwargs = dict(
            s_kernel_size=s_kernel_size,
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A_size,
        )
        self.stgc_block1 = STGCN.STGC_Block(
            config[0][0], config[0][1], config[0][2], **kwargs
        )
        self.stgc_block2 = STGCN.STGC_Block(
            config[1][0], config[1][1], config[1][2], **kwargs
        )
        self.stgc_block3 = STGCN.STGC_Block(
            config[2][0], config[2][1], config[2][2], **kwargs
        )

        # Prediction
        self.fc = nn.Conv2d(config[-1][1], num_classes, kernel_size=1, padding=0)

        # Attention
        self.att_bn = nn.BatchNorm2d(config[-1][1])
        self.att_conv = nn.Conv2d(
            config[-1][1], num_classes, kernel_size=1, padding=0, stride=1, bias=False
        )

        # Attention node
        self.att_node_conv = nn.Conv2d(
            num_classes, 1, kernel_size=1, padding=0, stride=1, bias=False
        )
        self.att_node_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        # Attention edge
        self.num_att_edge = num_att_edge
        self.att_edge_conv = nn.Conv2d(
            num_classes,
            num_att_edge * A_size[2],
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.att_edge_bn = nn.BatchNorm2d(num_att_edge * A_size[2])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, A):
        N, c, T, V = x.size()

        # STGC Block
        x = self.stgc_block1(x, A, None)
        x = self.stgc_block2(x, A, None)
        x = self.stgc_block3(x, A, None)

        # Prediction
        x_out = F.avg_pool2d(x, x.size()[2:])
        x_out = x_out.view(N, -1, 1, 1)
        x_out = self.fc(x_out)
        output = x_out.view(x_out.size(0), -1)

        # Attention
        x_att = self.att_bn(x)
        x_att = self.att_conv(x_att)

        # Attention node
        x_node = self.att_node_conv(x_att)
        x_node = self.att_node_bn(x_node)
        x_node = F.interpolate(x_node, size=(T, V))
        att_node = self.sigmoid(x_node)

        # Attention edge
        x_edge = F.avg_pool2d(x_att, (x_att.size()[2], 1))
        x_edge = self.att_edge_conv(x_edge)
        x_edge = self.att_edge_bn(x_edge)
        x_edge = x_edge.view(N, self.num_att_edge, V, V)
        x_edge = self.tanh(x_edge)
        att_edge = self.relu(x_edge)

        return output, att_node, att_edge


class FeatureExtractor(nn.Module):
    def __init__(self, config, s_kernel_size, t_kernel_size, dropout, A_size):
        super(FeatureExtractor, self).__init__()
        # Batch Normalization
        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])

        # STGC-Block config
        kwargs = dict(
            s_kernel_size=s_kernel_size,
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A_size,
        )
        self.stgc_block1 = STGCN.STGC_Block(
            config[0][0], config[0][1], config[0][2], **kwargs
        )
        self.stgc_block2 = STGCN.STGC_Block(
            config[1][0], config[1][1], config[1][2], **kwargs
        )
        self.stgc_block3 = STGCN.STGC_Block(
            config[2][0], config[2][1], config[2][2], **kwargs
        )

    def forward(self, x, A):
        # Batch Normalization
        N, C, T, V = x.size()  # batch, channel, frame, node
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        # STGC Blocks
        x = self.stgc_block1(x, A, None)
        x = self.stgc_block2(x, A, None)
        x = self.stgc_block3(x, A, None)
        return x
