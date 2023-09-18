import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        # self.bceloss = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')#F.binary_cross_entropy(inputs, targets, reduction='mean')
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # BCE = self.bceloss(inputs, targets)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, list_t, list_s):
        loss = 0
        for t, s in zip(list_t, list_s):
            loss += (1 - F.cosine_similarity(t, s)).mean()

        return loss

class CSAM(nn.Module):
    # cosine similarity attention module
    def __init__(self):
        super(CSAM, self).__init__()

    def forward(self, list_t, list_s):
        list_a = []
        for t, s in zip(list_t, list_s):
            att = (1 - F.cosine_similarity(t, s))/2
            att = torch.unsqueeze(att, dim=1)
            a = att*s
            list_a.append(a.detach())

        return list_a
        