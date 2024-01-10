import math
import pickle
import random
import sys
import time
from itertools import product

import h5py
import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary

from model import STGCN, Att_branch, Graph, Percep_branch
from utils import babel as datasets
from utils import log, util
from utils.opt import Options


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
    # print("xshape os", x.shape)
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

    x_out = x_out.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (m, seq_len, dim) * (m, seq_len, 1) = (m, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # weight is a gain parameter used to re-scale the standardized summed inputs
        # (dim) * (m, seq_len, dim) = (m, seq_Len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x, n_rep):
    # (batch_size, seq_len, n_kv_heads, head_dim, ?)([8, 2533, 8, 2, 2])
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
        keys, values = xk, xv
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
        scores = torch.matmul(scores, values)
        # print("SDPA output shape ", scores.shape)

        # ((m, n_heads, seq_len_q, head_dim) -> (m, seq_len_q, dim)
        output = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (m, seq_len_q, dim)
        return self.wo(output), scores


def sigmoid(x, beta=1):
    return 1 / (1 + torch.exp(-x * beta))


def swiglu(x, beta=1):
    return x * sigmoid(x, beta)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = 4 * config["embed_dim"]
        hidden_dim = int(2 * hidden_dim / 3)

        if config["ffn_dim_multiplier"] is not None:
            hidden_dim = int(config["ffn_dim_multiplier"] * hidden_dim)

        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = config["multiple_of"] * (
            (hidden_dim + config["multiple_of"] - 1) // config["multiple_of"]
        )

        self.w1 = nn.Linear(config["embed_dim"], hidden_dim, bias=False)
        self.w2 = nn.Linear(config["embed_dim"], hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config["embed_dim"], bias=False)

    def forward(self, x: torch.Tensor):
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        swish = swiglu(self.w1(x))
        # (m, seq_len, dim) --> (m, seq_len, hidden_dim)
        x_V = self.w2(x)

        # (m, seq_len, hidden_dim)
        x = swish * x_V

        # (m, seq_len, hidden_dim) --> (m, seq_len, dim)
        return self.w3(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config["n_heads"]
        self.dim = config["embed_dim"]
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

        # rms before attention block
        self.attention_norm = RMSNorm(self.dim, eps=config["norm_eps"])

        # rms before feed forward block
        self.ffn_norm = RMSNorm(self.dim, eps=config["norm_eps"])

    def forward(self, x, start_pos, freqs_complex):
        # (m, seq_len, dim)
        att_output, att_scores = self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        h = x + att_output
        # (m, seq_len, dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, att_scores


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.head_dim = config["embed_dim"] // config["n_heads"]
        self.num_classes = config["num_classes"]
        self.layers = nn.ModuleList()
        for layer_id in range(config["n_layers"]):
            self.layers.append(DecoderBlock(config))

        self.norm = RMSNorm(config["embed_dim"], eps=config["norm_eps"])
        self.output = nn.Linear(config["embed_dim"], self.num_classes, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.head_dim, config["max_seq_len"] * 2, device=(config["device"])
        )
        self.sublabel_fc1 = nn.Linear(
            config["max_seq_len"] * self.head_dim, config["max_seq_len"], bias=False
        )
        self.sublabel_fc2 = nn.Linear(
            config["max_seq_len"], config["max_seq_len"], bias=False
        )
        self.sublabel_fc3 = nn.Linear(
            config["max_seq_len"], config["num_subclasses"], bias=False
        )

        self.sublabelseg_fc1 = nn.Linear(
            self.head_dim, config["max_seq_len"] // 10, bias=False
        )
        self.sublabelseg_fc2 = nn.Linear(
            config["max_seq_len"] // 10, config["max_seq_len"], bias=False
        )
        self.sublabelseg_fc3 = nn.Linear(
            config["max_seq_len"], config["max_seq_len"], bias=False
        )
        self.sublabelseg_fc4 = nn.Linear(
            config["max_seq_len"], config["max_seq_len"], bias=False
        )

    def forward(self, tokens, start_pos):
        # (m, seq_len)
        batch_size, seq_len, d = tokens.shape

        # (m, seq_len) -> (m, seq_len, embed_dim)
        h = tokens

        # (seq_len, (embed_dim/n_heads)/2]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        # (m, seq_len, dim)
        for block_idx, layer in enumerate(self.layers):
            h, att_score = layer(h, start_pos, freqs_complex)
            if block_idx == 1:
                sublabel_att = att_score
        h = h.mean(dim=1)
        h = self.norm(h)

        # (m, seq_len, vocab_size)
        output = self.output(h).float()

        sublabel = sublabel_att.view(batch_size, self.n_heads, -1)
        sublabel = self.sublabel_fc1(sublabel)
        sublabel = self.sublabel_fc2(sublabel)
        sublabel = self.sublabel_fc3(sublabel)

        sublabelseg = self.sublabelseg_fc1(sublabel_att)
        sublabelseg = self.sublabelseg_fc2(sublabelseg)
        sublabelseg = self.sublabelseg_fc3(sublabelseg)
        sublabelseg = self.sublabelseg_fc4(sublabelseg)
        # print("sublabel shape", sublabel.shape)
        # softmax_result = F.softmax(output, dim=1)
        # # Get the result for each row (along axis 1)
        # output = softmax_result.argmax(dim=1)
        return (
            output,
            sublabel,
            sublabelseg,
        )


def main():
    BATCH_SIZE = 16
    NUM_EPOCH = 100

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    # データセットの用意
    # dataset = datasets.Datasets()
    # with open("dataset_babel_bmlmovi_120.pkl", "wb") as file:
    #     pickle.dump(dataset, file)

    data_loader = dict()

    with open("dataset_babel_bmlmovi_30.pkl", "rb") as file:
        dataset = pickle.load(file)

    # print("dataset shape is:", dataset.shape)
    print(">>> Training dataset length: {:d}".format(dataset.__len__()))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # print("training dataset shape is:", train_dataset.shape)

    data_loader["train"] = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    data_loader["test"] = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    max_len = data_loader["test"].sampler.data_source.dataset.data.shape[1]

    config = {
        "n_layers": 5,
        "embed_dim": 156,
        "n_heads": 13,
        "n_kv_heads": 13,
        "num_classes": 23,
        "multiple_of": 64,
        "ffn_dim_multiplier": None,
        "norm_eps": 1e-5,
        "max_batch_size": 8,
        "max_seq_len": max_len,
        "device": "cuda",
        "num_subclasses": 666,
    }

    print(
        "Input data shape is",
        input_shape := data_loader["test"].sampler.data_source.dataset.data.shape,
    )
    model = Transformer(config).to(config["device"])
    print(model)
    # res = model.forward(test_set["input_ids"].to(config["device"]), 0)
    # print(res.size())

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    # use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    # summary(model, (617, 156))
    # print(model)
    for epoch in range(1, NUM_EPOCH + 1):
        correct_pb = 0
        sum_loss = 0
        # IPython.embed()

        start_time = time.time()
        for batch_idx, (data, label, sublabel, sublabel_seg) in enumerate(
            data_loader["train"]
        ):
            data = data.cuda()
            label = label.cuda()
            sublabel = sublabel.cuda()
            sublabel_seg = sublabel_seg.cuda()
            sublabel = F.one_hot(sublabel, num_classes=config["num_subclasses"]).float()
            sublabel_seg_onehot = torch.zeros(
                sublabel_seg.shape[0],
                sublabel_seg.shape[1],
                config["max_seq_len"],
                config["max_seq_len"],
            ).cuda()
            for i, j in product(
                range(sublabel_seg.shape[0]), range(sublabel_seg.shape[1])
            ):
                if sublabel_seg[i, j, 1].item() > config["max_seq_len"] - 1:
                    sublabel_seg[i, j, 1] = config["max_seq_len"] - 1
                sublabel_seg_onehot[
                    i, j, sublabel_seg[i, j, 0].item(), sublabel_seg[i, j, 1].item()
                ] = 1
            output, output_sub, output_subseg = model(data, 0)
            # print("output_sub is", output_sub)
            # print("sublabel shape is", sublabel.shape)
            # print("output_sub shape is", output_sub.shape)
            loss = (
                criterion(output, label)
                + criterion(output_sub, sublabel)
                + criterion(output_subseg, sublabel_seg_onehot)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predict = torch.max(output.data, 1)
            correct_pb += (predict == label).sum().item()

        print(
            "estimated time left is {} mins".format(
                (time.time() - start_time) * (NUM_EPOCH - epoch) // 60
            ),
        )
        print(
            "# Epoch: {} | Loss: {:.4f} | Accuracy PB: {:.3f}[%]".format(
                epoch,
                sum_loss / len(data_loader["train"].dataset),
                (100.0 * correct_pb / len(data_loader["train"].dataset)),
            )
        )
        # if 100.0 * correct_pb / len(data_loader["train"].dataset) >= 20:
        #     break

    model.eval()

    correct_pb = 0
    with torch.no_grad():
        for batch_idx, (data, label, sublabel, sublabel_seg) in enumerate(
            data_loader["test"]
        ):
            data = data.cuda()
            label = label.cuda()
            tic = time.time()
            output, _, _ = model(data, 0)

            _, predict = torch.max(output.data, 1)
            correct_pb += (predict == label).sum().item()
            print("inference time is", time.time() - tic)

    print(
        "# Test Accuracy: {:.3f}[%]".format(
            100.0 * correct_pb / len(data_loader["test"].dataset)
        )
    )
    IPython.embed()


if __name__ == "__main__":
    # option = Options().parse()
    main()
