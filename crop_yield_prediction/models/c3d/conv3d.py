#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on code from https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/C3D_model.py
# Architecture is taken from https://esc.fnwi.uva.nl/thesis/centraal/files/f1570224447.pdf

import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, in_channels, n_tsteps):
        super(C3D, self).__init__()

        self.n_tsteps = n_tsteps

        # input (9, 7, 50, 50), output (9, 7, 50, 50)
        self.dimr1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn_dimr1 = nn.BatchNorm3d(in_channels, eps=1e-6, momentum=0.1)
        # output (3, 7, 50, 50)
        self.dimr2 = nn.Conv3d(in_channels, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.bn_dimr2 = nn.BatchNorm3d(3, eps=1e-6, momentum=0.1)

        # output (64, 7, 50, 50)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64, eps=1e-6, momentum=0.1)
        # output (64, 7, 25, 25)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # output (128, 7, 25, 25)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128, eps=1e-6, momentum=0.1)
        # output (128, 7, 12, 12)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # output (256, 7, 12, 12)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256, eps=1e-6, momentum=0.1)
        # output (256, 7, 12, 12)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256, eps=1e-6, momentum=0.1)
        # output (256, 3, 6, 6)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # output (512, 3, 6, 6)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512, eps=1e-6, momentum=0.1)
        # output (512, 3, 6, 6)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512, eps=1e-6, momentum=0.1)
        # output (512, 1, 3, 3)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool4_keept = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.fc5 = nn.Linear(4608, 1024)
        self.fc6 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):

        x = self.relu(self.bn_dimr1(self.dimr1(x)))
        x = self.relu(self.bn_dimr2(self.dimr2(x)))

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        if self.n_tsteps > 3:
            x = self.pool4(x)
        else:
            x = self.pool4_keept(x)

        # output (512, 1, 3, 3)
        x = x.view(-1, 4608)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)

        pred = torch.squeeze(self.fc6(x))

        return pred

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def get_1x_lr_params(model):
#     """
#     This generator returns all the parameters for conv and two fc layers of the net.
#     """
#     b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
#          model.conv5a, model.conv5b, model.fc6, model.fc7]
#     for i in range(len(b)):
#         for k in b[i].parameters():
#             if k.requires_grad:
#                 yield k
#
# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last fc layer of the net.
#     """
#     b = [model.fc8]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k

# if __name__ == "__main__":
#     inputs = torch.rand(1, 3, 16, 112, 112)
#     net = C3D(num_classes=101, pretrained=True)
#
#     outputs = net.forward(inputs)
#     print(outputs.size())
