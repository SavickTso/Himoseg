import math
from dataclasses import dataclass

# from utils_rf import scaled_dot_product, expand_mask
from typing import Optional, Tuple

import IPython
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import STGCN, utils_rf


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 1
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 64
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    # theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # (seq_len)
    m = torch.arange(seq_len, device=device)
    # (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # complex numbers in polar, c = R * exp(m * theta), where R = 1:
    # (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x, freqs_complex, device):
    # last dimension pairs of two values represent real and imaginary
    # two consecutive values will become a single complex number
    # (m, seq_len, num_heads, head_dim/2, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    # (m, seq_len, num_heads, head_dim/2)
    x_complex = torch.view_as_complex(x)
    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # multiply each complex number
    # (m, seq_len, n_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # convert back to the real number
    # (m, seq_len, n_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (m, seq_len, n_heads, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (m, seq_len, n_kv_heads, 1, head_dim)
        # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
        # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.dim = config["embed_dim"]
        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        self.n_heads_q = self.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x, start_pos, freqs_complex):
        # seq_len is always 1 during inference
        batch_size, seq_len, _ = x.shape

        # (m, seq_len, dim)
        xq = self.wq(x)

        # (m, seq_len, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (m, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (m, seq_len, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (m, seq_len, num_head, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)

        # (m, seq_len, h_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # (m, seq_len, h_kv, head_dim)
        keys = xk
        values = xv

        # (m, seq_len, h_kv, head_dim) --> (m, seq_len, n_heads, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (m, n_heads, seq_len, head_dim)
        # seq_len is 1 for xq during inference
        xq = xq.transpose(1, 2)

        # (m, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (m, n_heads, seq_len_q, head_dim) @ (m, n_heads, head_dim, seq_len) -> (m, n_heads, seq_len_q, seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # (m, n_heads, seq_len_q, seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (m, n_heads, seq_len_q, seq_len) @ (m, n_heads, seq_len, head_dim) -> (m, n_heads, seq_len_q, head_dim)
        output = torch.matmul(scores, values)

        # ((m, n_heads, seq_len_q, head_dim) -> (m, seq_len_q, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (m, seq_len_q, dim)
        return self.wo(output)


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
        # self.transformer_bn = nn.BatchNorm1d(num_features=52)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # Attention frame pairs
        # frame* frame
        # self.num_att_frame = 2533
        # self.att_frame_conv = nn.Conv2d(
        #     num_classes,
        #     self.num_att_frame * self.num_att_frame,
        #     kernel_size=1,
        #     padding=0,
        #     stride=1,
        #     bias=False,
        # )
        # self.att_frame_bn = nn.BatchNorm2d(self.num_att_frame**2)

        # Transformer
        self.positional_encoding = PositionalEncoding(20288)
        self.transformer1 = TransformerBlock(d_model=20288, n_heads=8, d_ff=128)

        # Multi-head self-attention encoder block x1
        # self.multi_head = EncoderBlock(64, 8, 64, dropout=0.2)

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
        x = x.permute(0, 3, 1, 2).contiguous().flatten(start_dim=2, end_dim=3)
        # IPython.embed()

        # self attention
        trans_x = self.positional_encoding(x)
        trans_x, att_mat = self.transformer1(trans_x)
        # x_att = x_att.permute(0, 3, 2, 1).contiguous()

        # attn_mtx = self.multi_head(x_att)
        # print("shape of attn_mtx: ", attn_mtx.shape)
        x_att = self.att_conv(x_att)

        # Attention node
        x_node = self.att_node_conv(x_att)
        x_node = self.att_node_bn(x_node)
        x_node = F.interpolate(x_node, size=(T, V))
        att_node = self.sigmoid(x_node)
        # print("attnodes are: ", att_node)

        # Attention edge
        x_edge = F.avg_pool2d(x_att, (x_att.size()[2], 1))
        x_edge = self.att_edge_conv(x_edge)

        x_edge = self.att_edge_bn(x_edge)
        x_edge = x_edge.view(N, self.num_att_edge, V, V)
        x_edge = self.tanh(x_edge)
        att_edge = self.relu(x_edge)
        # print("attedges are: ", att_edge)

        # Attention Frame
        # x_fpair = self.att_node_conv(x_att)
        # x_fpair = self.att_node_bn(x_fpair)
        # x_fpair = F.interpolate(x_fpair, size=(T, T))
        # att_frame = self.sigmoid(x_fpair)
        # print("attframes shape are: ", att_frame.shape)
        # print("attframes are: ", att_frame)

        return output, att_node, att_edge  # , attframe


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
