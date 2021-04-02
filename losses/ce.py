import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiCELoss(nn.Module):
    def __init__(self, weights = None, **kwargs):
        super().__init__()
        class_weights = torch.FloatTensor(weights).cuda()
        self.loss = nn.CrossEntropyLoss()
        #self.weights = weights

    def forward(self, output, target):
        # if self.weights is None:
        #     w = [1] * len(output)
        # else:
        #     w = self.weights
        #w = torch.ToTensor(w)
        # loss = 0.0
        output1 = torch.as_tensor(torch.cat(output, 0))
        target1 = torch.as_tensor(torch.cat(target, 0))

        # print(self.loss(output1, target1))
        # print(output1.shape)
        # print(target1.shape)
        # for i, (pred, true) in enumerate(zip(output, target)):
        #     loss += self.loss(pred, true)
        # print(loss / len(output))
        # exit(0)
        return self.loss(output1, target1)
