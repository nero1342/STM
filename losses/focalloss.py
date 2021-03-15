import torch.nn as nn 
import torch 


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = [0.25, 0.75], gamma = 2, size_average = True):
        super(FocalLoss, self).__init__() 
        self.num_classes = num_classes
        self.alpha = alpha 
        self.gamma = gamma 
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(alpha) == self.num_classes 
            self.alpha = torch.Tensor(list(self.alpha)) 
        elif isinstance(self.alpha, torch.Tensor):
            pass 
        else:
            raise TypeError("Not support alpha type")
        
    def forward(self, logit, target):
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.reshape(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.reshape(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        
        
        target = target.reshape(-1, 1) #[N, d1, d2, ...] => [N * d1 * d2 * .., 1]
        print(logit.shape, target.shape, logit.dim())

        # Memory saving way
        pt = logit.gather(1, target).view(-1) + self.eps 
        logpt = pt.log() 
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt 

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum() 
        return loss 

            
