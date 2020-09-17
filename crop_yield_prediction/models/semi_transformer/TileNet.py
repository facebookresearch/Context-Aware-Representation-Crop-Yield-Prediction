#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on tile2vec code from https://github.com/ermongroup/tile2vec

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 64

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4], stride=2)

    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, 4)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        return self.encode(x)

    def loss(self, anchor, temporal_neighbor, spatial_neighbor, spatial_distant, margin, l2, ltn):
        """
        Computes loss for each batch.
        """
        z_a, z_tn, z_sn, z_d = (self.encode(anchor), self.encode(temporal_neighbor),  self.encode(spatial_neighbor),
                                self.encode(spatial_distant))

        return triplet_loss(z_a, z_tn, z_sn, z_d, margin, l2, ltn)


def triplet_loss(z_a, z_tn, z_sn, z_d, margin, l2, ltn):
    dim = z_a.shape[-1]

    l_n = torch.sqrt(((z_a - z_sn) ** 2).sum(dim=1))
    l_d = - torch.sqrt(((z_a - z_d) ** 2).sum(dim=1))
    sn_loss = F.relu(l_n + l_d + margin)
    tn_loss = torch.sqrt(((z_a - z_tn) ** 2).sum(dim=1))

    # average by #samples in mini-batch
    l_n = torch.mean(l_n)
    l_d = torch.mean(l_d)
    l_nd = torch.mean(l_n + l_d)
    sn_loss = torch.mean(sn_loss)
    tn_loss = torch.mean(tn_loss)

    loss = (1 - ltn) * sn_loss + ltn * tn_loss

    norm_loss = 0
    if l2 != 0:
        z_a_norm = torch.sqrt((z_a ** 2).sum(dim=1))
        z_sn_norm = torch.sqrt((z_sn ** 2).sum(dim=1))
        z_d_norm = torch.sqrt((z_d ** 2).sum(dim=1))
        z_tn_norm = torch.sqrt((z_tn ** 2).sum(dim=1))
        norm_loss = torch.mean(z_a_norm + z_sn_norm + z_d_norm + z_tn_norm) / (dim ** 0.5)
        loss += l2 * norm_loss

    return loss, l_n, l_d, l_nd, sn_loss, tn_loss, norm_loss


def make_tilenet(in_channels, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim)

