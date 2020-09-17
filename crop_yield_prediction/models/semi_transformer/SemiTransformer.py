#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on transformer code from https://github.com/jadore801120/attention-is-all-you-need-pytorch

from crop_yield_prediction.models.semi_transformer.AttentionModels import Encoder
from crop_yield_prediction.models.semi_transformer.TileNet import make_tilenet

import torch
import torch.nn as nn


class SemiTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, tn_in_channels, tn_z_dim, tn_warm_start_model,
            sentence_embedding, output_pred, query_type,
            attn_n_tsteps, d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, apply_position_enc=True):

        super().__init__()

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        self.output_pred = output_pred

        self.tilenet = make_tilenet(tn_in_channels, tn_z_dim)

        self.encoder = Encoder(
            attn_n_tsteps, query_type=query_type, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
            apply_position_enc=apply_position_enc)

        self.sentence_embedding = sentence_embedding

        if self.output_pred:
            self.predict_proj = nn.Linear(d_model, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if tn_warm_start_model is not None:
            warm_start = torch.load(tn_warm_start_model)
            self.tilenet.load_state_dict(warm_start['model_state_dict'])

    def forward(self, x, unsup_weight):
        """
        Input x: (n_batches, n_tsteps, n_triplets, n_var, img_height, img_width)
        """
        n_batches, n_tsteps, n_triplets, n_vars, img_size = x.shape[:-1]

        emb_triplets = None
        if unsup_weight != 0:
            x = x.view(n_batches * n_tsteps * n_triplets, n_vars, img_size, img_size)

            emb_triplets = self.tilenet(x)

            emb_triplets = emb_triplets.view(n_batches, n_tsteps, n_triplets, -1)
            emb_x = emb_triplets[:, :, 0, :]
            # emb_triplets = emb_triplets.view(n_batches * n_tsteps, n_triplets, -1)
        else:
            x = x[:, :, 0, :, :, :]
            x = x.view(n_batches * n_tsteps, n_vars, img_size, img_size)
            emb_x = self.tilenet(x)
            emb_x = emb_x.view(n_batches, n_tsteps, -1)

        enc_output, *_ = self.encoder(emb_x)

        if self.sentence_embedding == 'simple_average':
            enc_output = enc_output.mean(1)

        pred = torch.squeeze(self.predict_proj(enc_output))

        return emb_triplets, pred
