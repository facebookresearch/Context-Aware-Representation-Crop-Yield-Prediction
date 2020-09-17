#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from crop_yield_prediction.models.semi_transformer.TileNet import make_tilenet

import torch
import torch.nn as nn


class CnnLstm(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, tn_in_channels, tn_z_dim, d_model=512, d_inner=2048):

        super().__init__()

        self.tilenet = make_tilenet(tn_in_channels, tn_z_dim)

        self.encoder = nn.LSTM(d_model, d_inner, batch_first=True)

        self.predict_proj = nn.Linear(d_inner, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Input x: (n_batches, n_tsteps, n_triplets, n_var, img_height, img_width)
        """
        n_batches, n_tsteps, n_vars, img_size = x.shape[:-1]

        x = x.view(n_batches * n_tsteps, n_vars, img_size, img_size)
        emb_x = self.tilenet(x)
        emb_x = emb_x.view(n_batches, n_tsteps, -1)

        enc_output, *_ = self.encoder(emb_x)
        enc_output = enc_output[:, -1, :]

        pred = torch.squeeze(self.predict_proj(enc_output))

        return pred
