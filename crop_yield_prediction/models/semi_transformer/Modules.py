#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, gq=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        if gq is not None:
            attn_gq = torch.matmul(gq / self.temperature, k.transpose(2, 3))
            attn_gq = self.dropout(F.softmax(attn_gq, dim=-1))
            attn += attn_gq
        output = torch.matmul(attn, v)

        return output, attn
