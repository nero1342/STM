import torch
import torch.nn.functional as F

def multi_class_prediction(output):
    return torch.argmax(output, dim=1)


def binary_prediction(output, thresh=0.0):
    return (output.squeeze(1) > thresh).long()

class MeanIoU():
    def __init__(self, ignore_index=None, eps=1e-6):
        super().__init__()
        self.pred_fn = multi_class_prediction
        
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def calculate(self, output, target):
        batch_size = output.size(0)
        
        #ious = torch.zeros((batch_size, nclasses))

        prediction = self.pred_fn(output)
        nclasses = int(max(torch.max(prediction), torch.max(target)) + 1)
        if self.ignore_index is not None:
            target_mask = (target == self.ignore_index).bool()
            prediction[target_mask] = self.ignore_index

        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        ious = (intersection.float() + self.eps) / (union.float() + self.eps)

        return ious.cpu()

    def update(self, value):
        self.mean_class += value.sum()
        self.sample_size += value.size(1)

    def value(self):
        return self.mean_class / self.sample_size

    def reset(self):
        self.mean_class = 0
        self.sample_size = 0

    def summary(self):
        print(self.sample_size)
        # class_iou = self.mean_class / self.sample_size

        print(f'mIoU: {self.value():.6f}')
        # for i, x in enumerate(class_iou):
        #     print(f'\tClass {i:3d}: {x:.6f}')

class ModifiedMeanIoU(MeanIoU):
    def calculate(self, output, target):
        return super().calculate(output[-1], target[-1])