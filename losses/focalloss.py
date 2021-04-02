import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

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
        
        print(self.alpha)
    def forward(self, logit, target):
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.reshape(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.reshape(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        
        
        target = target.reshape(-1, 1) #[N, d1, d2, ...] => [N * d1 * d2 * .., 1]
        #print(logit.shape, target.shape, logit.dim())

        # Memory saving way
        pt = logit.gather(1, target).view(-1) + self.eps 
        logpt = pt.log() 
        print(logpt[:2])
        print((torch.pow(1 - pt, self.gamma))[:2])
        print((torch.pow(1 - pt, self.gamma) * logpt)[:2])
        print()
        # exit(0)
        loss = -1 * torch.pow(1 - pt, self.gamma) * logpt 
        print(loss[:2])
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum() 
        return loss 

            
class FocalLossV2(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    Usage is like this:
        >>> criteria = FocalLossV2()
        >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        >>> loss = criteria(logits, lbs)
    '''
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.vars = (coeff, probs, log_probs, log_1_probs, probs_gamma,
                probs_1_gamma, label, gamma)

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, log_probs, log_1_probs, probs_gamma,
                probs_1_gamma, label, gamma) = ctx.vars

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None