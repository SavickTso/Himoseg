import math
import torch
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
