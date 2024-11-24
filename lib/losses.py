import torch
import numpy as np
import torch.nn.functional as F

def loss_select(name):
    if name.startswith("CSI"):
        return MaskedMSELoss
    elif name == 'StockMixerLoss':
        return StockMixerLoss
    else:
        raise NotImplementedError
    
def masked_mse_loss(pred, label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss = torch.mean((label[mask] - pred[mask])**2)
    return torch.mean(loss)

## need update
def masked_ranking_loss(pred, label,null_val=0.0):
    device = torch.device('cuda')
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    gt_diff = label.unsqueeze(1) - label.unsqueeze(0)

    loss_matrix = F.relu(-gt_diff * pred_diff)
    rank_loss = loss_matrix.sum() / (pred.size(0) ** 2) 
    return rank_loss
    
    


class MaskedMSELoss(torch.nn.Module):
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self,pred,label,null_val=0.0):
        return masked_mse_loss(pred,label,null_val=null_val)

class StockMixerLoss(torch.nn.Module):
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self,pred,label,alpha=0.1,null_val=0.0):
        return masked_mse_loss(pred,label,null_val=null_val) + alpha * masked_ranking_loss(pred,label,null_val=null_val)