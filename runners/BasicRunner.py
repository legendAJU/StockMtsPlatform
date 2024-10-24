import pandas as pd
import numpy as np
import torch
import sys
from torch import nn
from .AbstractRunner import AbstractRunner
from lib.metrics import IC_RIC
from lib.utils import print_log
sys.path.append("..")


class BasicRunner(AbstractRunner):
    def __init__(self,cfg:dict, device,log=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.log = log
        self.clip_grad = cfg['clip_grad']
        print(self.clip_grad)
    def train_one_epoch(self, model,train_loader,optimizer,criterion):
        model.train()
        train_losses = []
        for data in train_loader:
            data = torch.squeeze(data,dim=0)
            feature = data[:,:,0:-1].to(self.device)
            label = data[:,-1,-1].to(self.device)
            # forward
            pred = model(feature.float())
            loss = criterion(pred,label)
            train_losses.append(loss.item())
            #back
            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(),self.clip_grad)
            optimizer.step()
        epoch_loss = float(np.mean(train_losses))
        return epoch_loss
    def eval_model(self,model,valid_loader,criterion):
        model.eval()
        valid_losses = []
        for data in valid_loader:
            data = torch.squeeze(data,dim=0)
            feature = data[:,:,0:-1].to(self.device)
            label = data[:,-1,-1].to(self.device)
            # forward
            pred = model(feature.float())
            loss = criterion(pred,label)
            valid_losses.append(loss.item())
        valid_loss = float(np.mean(valid_losses))
        return valid_loss
    def predict(self,model,test_loader):
        model.eval()
        ic = []
        ric = []
        for data in test_loader:
            data = torch.squeeze(data,dim=0)
            feature = data[:,:,0:-1].to(self.device)
            label = data[:,-1,-1]
            # forward
            pred = model(feature.float()).detach().cpu().numpy()
            daily_ic,daily_ric = IC_RIC(label.detach().numpy(),pred)
            ic.append(daily_ic)
            ric.append(daily_ric)
        metrics = {
            'IC' : np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC' : np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)  
        }
        return metrics
    def train(self,model,train_loader,epoches,optimizer,criterion):
        model.train()
        total_losses = []
        for epoch in range(epoches):
            train_loss = self.train_one_epoch(model,train_loader,optimizer,criterion)
            total_losses.append(train_loss)
            print_log(f"Epoch {epoch} Train Loss: {train_loss}",log=self.log)
 