import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import pandas as pd
import numpy as np


class DataSet(Dataset):
    def __init__(self,name):
        self.name = name.upper()
        if(name == "CSI300"):
            with open('../data/csi300/csi300_dl_train.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open('../data/csi300/csi300_dl_valid.pkl', 'rb') as f:
                self.valid_data = pickle.load(f)
            with open('../data/csi300/csi300_dl_test.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
        elif(name == "CSI800"):
            with open('../data/csi800/csi800_dl_train.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open('../data/csi800/csi800_dl_valid.pkl', 'rb') as f:
                self.valid_data = pickle.load(f)
            with open('/..data/csi800/csi800_dl_test.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
        #elif(name == 'NASDAQ'):
    
    def load_csi_data(self,shuffle_train):
        train_loader = init__data_loader(self.train_data,shuffle=shuffle_train,drop_last=True)
        valid_loader = init__data_loader(self.valid_data,shuffle=False,drop_last=True)
        test_loader = init__data_loader(self.test_data,shuffle=False,drop_last=True)
        return train_loader,valid_loader,test_loader
    
    def load_csi_data_batched(self,batch_size,shuffle_train):
        self.train_data.config(fillna_type="ffill+bfill")  
        self.valid_data.config(fillna_type="ffill+bfill")
        self.test_data.config(fillna_type="ffill+bfill")
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle_train,drop_last=True)
        valid_loader = DataLoader(self.valid_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = init__data_loader(self.test_data, shuffle=False,drop_last=True)
        return train_loader, valid_loader,test_loader

    def data_loader_select(self,model_name):
        if(self.name.startswith("CSI")):
            if model_name in ("MASTER","GATS","GAT"):
                return self.load_csi_data
            elif model_name in("LSTM","TRANSFORMER","GRU"):
                return self.load_csi_data_batched
    
    def __len__(self):
        return len(self.train_data)


def init__data_loader(train_data,shuffle=True,drop_last=True):
    sampler = DailyBatchSamplerRandom(train_data,shuffle)
    data_loader = DataLoader(train_data,sampler = sampler,drop_last=drop_last)
    return data_loader

class DailyBatchSamplerRandom(Sampler):
    def __init__(self,data_source,shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

        self.daily_count = pd.Series(index = data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count),1)
        self.daily_index[0] = 0
    def __iter__(self):
        if self.shuffle:
            #print("Start Sampling...")
            index = np.arange(len(self.daily_index))
            #print("Daily Index:",self.daily_index)
            np.random.shuffle(self.daily_index)
            #print("Shuffled Daily Index:",self.daily_index)
            for i in index:
                #print("Daily Index:",self.daily_index[i],self.daily_index[i] + self.daily_count[i])
                yield np.arange(self.daily_index[i],self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)
    def __len__(self):
        return len(self.data_source)

    