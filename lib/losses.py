import torch
import numpy as np
import torch.nn.functional as F

def loss_select(name):
    if name.startswith("CSI"):
        return MaskedMSELoss
    
def masked_mse_loss(pred, label,null_val=0.0):
    mask = ~torch.isnan(label)
    loss = torch.mean((label[mask] - pred[mask])**2)
    return torch.mean(loss)

## need update
def masked_ranking_loss(pred, label,null_val=0.0):
    mask = ~torch.isnan(label)
    device = pred.device()
    all_one = torch.ones(len(pred),1,dtype=torch.float32).to(device)
    pre_pw_dif = torch.sub(
        pred @ all_one.t(),
        all_one @ pred.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ label.t(),
        label @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    return rank_loss
    
    


class MaskedMSELoss(torch.nn.Module):
    def _get_name(self):
        return self.__class__.__name__
    
    def __call__(self,pred,label,null_val=0.0):
        return masked_mse_loss(pred,label,null_val=null_val)