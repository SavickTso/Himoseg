import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython
from model import utils_rf
import math
import os, sys
from sys import exit
from utils import amass as datasets
from torch.utils.data import DataLoader, random_split

# from utils_rf import scaled_dot_product, expand_mask


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        # print(x.size())
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = utils_rf.expand_mask(mask)
        qkv = self.qkv_proj(x)
        # print("the shape of qkv", qkv.shape)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = utils_rf.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class full_transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        dropout=0.0,
        input_dropout=0.0,
    ):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout), nn.Linear(input_dim, model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.input_net(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)

        return x


def main():
    BATCH_SIZE = 8
    NUM_EPOCH = 100

    model = full_transformer(
        input_dim=156, model_dim=16, num_classes=23, num_heads=8, num_layers=64
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 誤差関数
    criterion = torch.nn.CrossEntropyLoss()

    # データセットの用意
    data_loader = dict()
    dataset = datasets.Datasets()
    # IPython.embed()
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(">>> Training dataset length: {:d}".format(dataset.__len__()))
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

    model.train()
    print(model)

    for epoch in range(1, NUM_EPOCH + 1):
        correct_pb = 0
        sum_loss = 0
        # IPython.embed()
        for batch_idx, (data, label) in enumerate(data_loader["train"]):
            data = data.cuda()
            label = torch.eye(23)[label].long()  # numclasses=23
            label = label.cuda()
            # print(batch_idx)
            # print("labels shape", label.shape)
            # print("labels shape", label)
            output_pb = model(data)
            print("the output shape is", output_pb.shape)
            print("the output is", output_pb)
            loss = criterion(output_pb, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            _, predict = torch.max(output_pb.data, 1)
            correct_pb += (predict == label).sum().item()

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


if __name__ == "__main__":
    main()
