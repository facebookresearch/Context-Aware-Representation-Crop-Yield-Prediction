#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

from crop_yield_prediction.models.semi_transformer.Layers import EncoderLayer

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(n_position, d_hid)
        position = torch.arange(0.0, n_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_hid, 2) * -(math.log(10000.0) / d_hid))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_tsteps, query_type, d_word_vec, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout=0.1,
            apply_position_enc=True):

        super().__init__()

        self.apply_position_enc = apply_position_enc

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_tsteps)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_tsteps, query_type, d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        if self.apply_position_enc:
            x = self.position_enc(x)
        enc_output = self.dropout(x)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,
