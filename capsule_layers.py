# -*- coding: utf-8 -*-
"""
@Time   : 2017/12/22 17:09
@Author : Nico
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm)


class CapsuleHidden(nn.Module):
    def __init__(self, num_inputs, num_outputs, vec_len,
                 kernel_size=9, stride=2):
        super(CapsuleHidden, self).__init__()
        self.vec_len = vec_len
        self.capsules = nn.ModuleList(
            [nn.Conv2d(num_inputs, num_outputs, kernel_size, stride) for _ in xrange(vec_len)]
        )

    def forward(self, inputs):
        # input shape [N, 256, 20, 20]
        outputs = [capsule(inputs).view(inputs.size(0), -1, 1) for capsule in self.capsules]
        # shape 8 * [N, 6*6*32, 1]
        outputs = torch.cat(outputs, dim=-1)
        # shape [N, 1152, 8]
        outputs = squash(outputs)
        return outputs


class CapsuleOut(nn.Module):
    def __init__(self, num_inputs, num_outputs, vec_len,
                 num_route_nodes, num_iterations):
        super(CapsuleOut, self).__init__()
        # shape [10, 1152, 8, 16]
        self.route_weights = nn.Parameter(torch.randn(num_outputs, num_route_nodes, num_inputs, vec_len))
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

    def forward(self, inputs):
        # input shape [N, 1152, 8] to [N, 1, 1152, 1, 8]
        inputs = inputs[:, None, :, None, :]
        # shape [1, 10, 1152, 8, 16]
        weights = self.route_weights[None, ...]
        # shape [N, 10, 1152, 1, 16]
        priors = torch.matmul(inputs, weights)
        logits = Variable(torch.zeros(priors.shape)).cuda()
        for i in xrange(self.num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = squash(torch.sum(probs * priors, dim=2, keepdim=True))
            if i < self.num_iterations:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits
        return outputs


def capsule_loss(images, labels, classes, reconstructions):
    left = F.relu(0.9 - classes, inplace=True) ** 2
    right = F.relu(classes - 0.1, inplace=True) ** 2

    margin_loss = labels * left + 0.5 * (1. - labels) * right
    margin_loss = margin_loss.sum()

    reconstruction_loss = nn.MSELoss(size_average=False)(reconstructions, images)

    return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
