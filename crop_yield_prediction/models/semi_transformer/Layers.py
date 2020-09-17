#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

from crop_yield_prediction.models.semi_transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

import torch.nn as nn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_tsteps, query_type, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_tsteps, query_type, n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
