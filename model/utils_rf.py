import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def eliminate_frames_4d(tensor_4d, desired_length):
    """For least relative node/frame elimination"""
    # Sum all frames' values both horizontally and vertically
    # frame_eliminated = []
    sum_list = (
        torch.sum(tensor_4d, dim=-1)
        + torch.sum(tensor_4d, dim=-2)
        - torch.diagonal(tensor_4d, dim1=-1, dim2=-2)
    )
    # keep_indices = torch.tensor(range(100))[~torch.isin(torch.tensor(range(100)), eliminate_indices)]
    keep_indices = torch.argsort(sum_list)[:, :, -desired_length:]
    eliminate_indices = torch.argsort(sum_list)[:, :, :-desired_length]
    print("length of eliminate frames are {}".format(eliminate_indices.shape[2]))
    tensor_eliminated = torch.Tensor(
        tensor_4d.shape[0], tensor_4d.shape[1], desired_length, desired_length
    ).cuda()
    # Add : later for dimension format
    # print("keepindices.shape", keep_indices)
    for i in range(tensor_4d.shape[0]):
        for j in range(tensor_4d.shape[1]):
            # print(keep_indices[i][j])
            keep_indices[i][j], _ = torch.sort(keep_indices[i][j])
            # print(keep_indices[i][j])

            tensor_eliminated[i][j] = tensor_4d[i][j][keep_indices[i][j]][
                :, keep_indices[i][j]
            ]
    # print("tensor_eliminated.shape is ", tensor_eliminated.shape)
    return tensor_eliminated


def topk_selection(m_1d, k, t_len):
    """
    Parameters:
    - m_1d (tensor): the input 1-dimention matrix.
    - k (int): The number of candidates.
    - t_len (int): The length of the iput sequence.
    """

    flattened_scores, indices = torch.topk(m_1d.view(-1), k)
    high_score_pairs = [(i // t_len, i % t_len) for i in indices]

    return high_score_pairs


def threshold_selection(m_1d, threshold):
    """
    Parameters:
    - m_1d (tensor): the input 1-dimention matrix.
    - threshold (float): The value of threshold.
    """
    high_score_pairs = (m_1d > threshold).nonzero().tolist()

    return high_score_pairs


def expand_mask(mask):
    """
    Helper function to support different mask shapes.
    Output shape supports (batch_size, number of heads, seq length, seq length)
    If 2D: broadcasted over batch size and number of heads
    If 3D: broadcasted over number of heads
    If 4D: leave as is
    """
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


def scaled_dot_product(q, k, v, mask=None):
    """
    Classic self-attention mechanism
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    # print(attn_logits)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
