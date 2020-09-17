#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

''' Define the sublayers in encoder/decoder layer '''
from crop_yield_prediction.models.semi_transformer.Modules import ScaledDotProductAttention

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_tsteps, query_type, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.global_query = nn.Parameter(torch.randn(n_head, d_k, n_tsteps), requires_grad=True)
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.query_type = query_type
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # sz_b: batch size, len_q, len_k, len_v: number of time steps
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if self.query_type == 'global':
            q = self.global_query
            q = q.transpose(1, 2)  # transpose to n * lq * dk
        elif self.query_type == 'fixed':
            q = self.layer_norm(q)
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            q = q.transpose(1, 2)  # transpose to b x n x lq x dk
        elif self.query_type == 'combine':
            lq = self.layer_norm(q)
            lq = self.w_qs(lq).view(sz_b, len_q, n_head, d_k)
            lq = lq.transpose(1, 2)
            gq = self.global_query
            gq = gq.transpose(1, 2)
            q = lq + gq
        elif self.query_type == 'separate':
            lq = self.layer_norm(q)
            lq = self.w_qs(lq).view(sz_b, len_q, n_head, d_k)
            lq = lq.transpose(1, 2)
            gq = self.global_query
            gq = gq.transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        k, v = k.transpose(1, 2), v.transpose(1, 2)  # Transpose for attention dot product: b x n x lq x dv
        # Transpose for attention dot product: b x n x lq x dv
        if self.query_type == 'separate':
            q, attn = self.attention(lq, k, v, gq)
        else:
            q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x
