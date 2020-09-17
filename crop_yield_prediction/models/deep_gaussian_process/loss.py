#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapt code from https://github.com/gabrieltseng/pycrop-yield-prediction

import torch.nn.functional as F


def l1_l2_loss(pred, true, l1_weight, scores_dict):
    """
    Regularized MSE loss; l2 loss with l1 loss too.

    Parameters
    ----------
    pred: torch.floatTensor
        The model predictions
    true: torch.floatTensor
        The true values
    l1_weight: int
        The value by which to weight the l1 loss
    scores_dict: defaultdict(list)
        A dict to which scores can be appended.

    Returns
    ----------
    loss: the regularized mse loss
    """
    loss = F.mse_loss(pred, true)

    scores_dict['l2'].append(loss.item())

    if l1_weight > 0:
        l1 = F.l1_loss(pred, true)
        loss += l1
        scores_dict['l1'].append(l1.item())
    scores_dict['loss'].append(loss.item())

    return loss, scores_dict
