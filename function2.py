import os

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.nn import ReLU, Module, Linear, Sigmoid, BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torch import Tensor
from sklearn.metrics import accuracy_score

Image_folder = 'images/Images'
BATCH_SIZE = 64
SHAPE = (224,224)

class DogDataset(Dataset):
    """
    X = [paths to images]
    y = [class ids]
    class_names = [class name]
    """
    def __init__(self,path,num_class=120,*,one_hot=1):#path = ".../Images/"
        #load the path only to reduce runtime
        self.class_names = []
        self.num_class = num_class
        self.X = []
        self.y = []
        current_class = 0
        for dirname,_, filenames in list(os.walk(path))[1:num_class+1]:
            class_name = dirname.split('/')[-1].split('-')[-1]
            self.class_names.append(class_name)
            self.X.extend(os.path.join(dirname,filename)for filename in filenames)
            self.y.extend([current_class]*len(filenames))
            current_class += 1
        if one_hot:#120 class y[i]=3 -> [0,0,0,1, ...0]
            for _ in range(len(self.y)):
                tmp = [0]*self.num_class
                tmp[self.y[_]] = 1
                self.y[_] = tmp
        self.y = np.array(self.y,dtype = 'float32')
        self.y = torch.tensor(self.y,dtype= torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]
    def get_splits(self,n_test=0.33):
        test_size = int(len(self.X)*n_test)
        return random_split(self,[len(self.X)-test_size,test_size])
    
if __name__=='__main__':
    dataset = DogDataset(Image_folder,one_hot=0)
    print(dataset.class_names)
    train_set, test_set = dataset.get_splits()
    print(train_set)
    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
