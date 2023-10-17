import torch.nn as nn
import torch.nn.functional as F
from model import STGCN


class PerceptionBranch(nn.Module):
    def __init__(
        self,
        config,
        num_classes,
        num_att_edge,
        s_kernel_size,
        t_kernel_size,
        dropout,
        A_size,
        use_att_edge=True,
    ):
        super(PerceptionBranch, self).__init__()
        # STGC-Block config
        kwargs = dict(
            s_kernel_size=s_kernel_size,
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A_size,
            num_att_edge=num_att_edge,
            use_att_edge=use_att_edge,
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

    def forward(self, x, A, att_edge):
        N, c, T, V = x.size()
        # STGC Block
        x = self.stgc_block1(x, A, att_edge)
        x = self.stgc_block2(x, A, att_edge)
        x = self.stgc_block3(x, A, att_edge)

        # Prediction
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)
        x = self.fc(x)
        output = x.view(x.size(0), -1)

        return output
