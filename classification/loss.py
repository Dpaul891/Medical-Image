import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, weight=None):
        '''
        Initialize some essential hyperparameters for your loss function
        '''
        super(MyLoss, self).__init__()
        ### Begin your code ###
        if weight is None:
            weight = [1.0, 1.5, 2.0]
        self.weight = weight
        ### End your code ###
    
    def forward(self, outputs, labels):
        '''
        Define the calculation of the loss
        '''
        ### Begin your code ###
        labels = labels.flatten()
        outputs = F.log_softmax(outputs, dim=-1)

        loss = 0
        for i in range(3):
            loss += - 1/labels.shape[0] *outputs[labels == i, i].sum() * self.weight[i]

        return loss
        ### End your code ###