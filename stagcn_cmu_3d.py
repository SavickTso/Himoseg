from utils import amass as datasets
from model import Graph, Percep_branch, STGCN, Att_branch
from utils.opt import Options
from utils import util
from utils import log

import IPython
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import h5py
import torch.optim as optim


class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        super().__init__()
        self.label = np.load(label_path)
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data = np.array(self.data[index])
        label = self.label[index]

        return data, label


class STA_GCN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        t_kernel_size,
        hop_size,
        num_att_edge,
        dropout=0.5,
    ):
        super(STA_GCN, self).__init__()

        # Graph
        graph = Graph.Graph(hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        kwargs = dict(
            s_kernel_size=A.size(0),
            t_kernel_size=t_kernel_size,
            dropout=dropout,
            A_size=A.size(),
        )

        # Feature extractor
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1]]
        self.feature_extractor = Att_branch.FeatureExtractor(f_config, **kwargs)

        # Attention branch
        a_config = [[32, 64, 2], [64, 64, 1], [64, 64, 1]]
        self.attention_branch = Att_branch.AttentionBranch(
            a_config, num_classes, num_att_edge, **kwargs
        )

        # Perception branch
        p_config = [[32, 64, 2], [64, 64, 1], [64, 64, 1]]
        self.perception_branch = Percep_branch.PerceptionBranch(
            p_config, num_classes, num_att_edge, **kwargs
        )

    def forward(self, x):
        # Feature extractor
        feature = self.feature_extractor(x, self.A)

        # Attention branch
        output_ab, att_node, att_edge = self.attention_branch(feature, self.A)

        # Attention mechanism
        att_x = feature * att_node

        # Perception branch
        output_pb = self.perception_branch(att_x, self.A, att_edge)

        return output_ab, output_pb, att_node, att_edge


def main():
    NUM_EPOCH = 100
    BATCH_SIZE = 64
    HOP_SIZE = 2
    NUM_ATT_EDGE = 2  # 動作ごとのattention edgeの生成数

    # モデルを作成
    model = STA_GCN(
        num_classes=10,
        in_channels=3,
        t_kernel_size=9,  # 時間グラフ畳み込みのカーネルサイズ (t_kernel_size × 1)
        hop_size=HOP_SIZE,
        num_att_edge=NUM_ATT_EDGE,
    ).cuda()

    # オプティマイザ
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 誤差関数
    criterion = torch.nn.CrossEntropyLoss()

    # データセットの用意
    data_loader = dict()
    dataset = datasets.Datasets(split=0)
    print(">>> Training dataset length: {:d}".format(dataset.__len__()))
    data_loader["train"] = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    # valid_dataset = datasets.Datasets(1, split=1)
    # print(">>> Validation dataset length: {:d}".format(valid_dataset.__len__()))
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=False,
    # )

    test_dataset = datasets.Datasets(1, split=2)
    print(">>> Testing dataset length: {:d}".format(test_dataset.__len__()))
    data_loader["test"] = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("so far so good")
    # data_loader = dict()
    # data_loader["train"] = torch.utils.data.DataLoader(
    #     dataset=Feeder(
    #         data_path="data/train_data.npy", label_path="data/train_label.npy"
    #     ),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    # )
    # data_loader["test"] = torch.utils.data.DataLoader(
    #     dataset=Feeder(
    #         data_path="data/test_data.npy", label_path="data/test_label.npy"
    #     ),
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    # )

    # モデルを学習モードに変更
    model.train()

    # 学習開始
    for epoch in range(1, NUM_EPOCH + 1):
        correct_pb = 0
        sum_loss = 0
        IPython.embed()
        for batch_idx, (data, label) in enumerate(data_loader["train"]):
            print("still not bad")
            data = data.cuda()
            label = label.cuda()

            output_ab, output_pb, _, _ = model(data)

            loss = criterion(output_ab, label) + criterion(output_pb, label)
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


if __name__ == "__main__":
    # option = Options().parse()
    main()
