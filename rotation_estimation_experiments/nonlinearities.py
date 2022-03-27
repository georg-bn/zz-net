# -*- coding: utf-8 -*-

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import init



class RadRelu(torch.nn.Module):

    """
        The radial relu.
            x -> pos(|x|-t)/|x|*x

        Input: point cloud (batch_size, n_in,m,d)
        Output: point cloud (batch_size, n_in,m,d)
        #Input:  (batch_size, n_in, d)
        #Output: (batch_size, n_in, d)

        Input parameters:
            n_in - number of point clouds
            tau - initial thres-value

        Attributes:
            thres - the ReLU-cutoff point
    """
    def __init__(self, n_in, tau=.5):
        super(RadRelu, self).__init__()
        self.thres = torch.nn.Parameter(torch.empty((1, n_in),
                                                    dtype=torch.float))
        self.reset_parameters(tau)

    def reset_parameters(self, tau):
        init.constant_(self.thres, tau)

    def forward(self, input):

        norms = torch.sqrt(((input**2).sum(3))) + 1e-7

        input = input/norms.unsqueeze(3)

        # input[input.isnan()] = 0

        return F.relu(norms-self.thres[:, :, None])[:, :, :, None]*input

