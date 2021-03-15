import torch


class Accuracy():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f'Accuracy: {self.value()}')


import numpy as np 

class AverageAccuracy():
    def __init__(self, nclasses, **kwargs):
        self.nclasses = nclasses
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        pred[pred != target] = self.nclasses
        
        freq = torch.bincount(pred, minlength=self.nclasses)
        freq_tar = torch.bincount(target, minlength=self.nclasses)
        return (freq, freq_tar) #correct, sample_size

    def update(self, value):
        freq, freq_tar = value 
        self.correct += freq.cpu().numpy()[:self.nclasses]
        self.sample_size += freq_tar.cpu().numpy()[:self.nclasses]

    def reset(self):
        self.correct = np.zeros(self.nclasses)
        self.sample_size = np.zeros(self.nclasses)


    def value(self):
        return np.mean(self.correct / self.sample_size)

    def summary(self):
        print(f'Accuracy: {self.value()}\n {self.correct}\n {self.sample_size}')