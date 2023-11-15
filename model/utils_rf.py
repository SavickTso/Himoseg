import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
