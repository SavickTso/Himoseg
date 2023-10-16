import numpy as np
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import torch.nn.functional as F


class Graph:
    def __init__(self, hop_size=2):
        self.get_edge()
        self.hop_size = hop_size
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, hop_size=hop_size
        )
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 52
        self_link = [(i, i) for i in range(self.num_node)]  # ループ
        neighbor_base = [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 6),
            (4, 7),
            (5, 8),
            (6, 9),
            (7, 10),
            (8, 11),
            (9, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (13, 16),
            (14, 17),
            (15, 18),
            (17, 19),
            (18, 20),
            (19, 21),
            (20, 22),
            (21, 23),
            (23, 24),
            (24, 25),
            (21, 26),
            (26, 27),
            (27, 28),
            (21, 29),
            (29, 30),
            (30, 31),
            (21, 32),
            (32, 33),
            (33, 34),
            (21, 35),
            (35, 36),
            (36, 37),
            (22, 38),
            (38, 39),
            (39, 40),
            (22, 41),
            (41, 42),
            (42, 43),
            (22, 44),
            (44, 45),
            (45, 46),
            (22, 47),
            (47, 48),
            (48, 49),
            (22, 50),
            (50, 51),
            (51, 52),
        ]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(A, Dn)
        return DAD
