import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        w = 1.0 / len(output)
        loss = 0.0
        for pred, true in zip(output, target):
            loss += self.loss(pred, true) * w
        return loss
