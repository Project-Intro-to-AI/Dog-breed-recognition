import os

import numpy as np
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

class DogDataset(Dataset):
    """
    X = [paths to images]
    y = [class ids]
    class_names = [class name]
    """
    def __init__(self,path,num_class=120,*,one_hot=1):#path = ".../Images/"
        #load the path only to reduce runtime
        self.class_names = [] # Danh sách tên các class
        self.num_class = num_class 
        self.X = [] #Danh sách chứa đường dẫn đến tất cả các ảnh.
        self.y = [] #Danh sách chứa label (id class) tương ứng với mỗi ảnh.
        current_class = 0 #Biến đếm class hiện tại

        for dirname,_, filenames in list(os.walk(path))[1:num_class+1]: #Duyệt qua các thư mục con trong thư mục path (bỏ qua thư mục gốc).
            # dirname: Đường dẫn của thư mục con hiện tại
            # _: Danh sách các thư mục con
            # filenames: danh sách các file có trong thư mục con
            class_name = dirname.split('/')[-1].split('-')[-1] # Tách tên class từ đường dẫn thư mục con   
            self.class_names.append(class_name)
            self.X.extend(os.path.join(dirname,filename)for filename in filenames) #Thêm đường dẫn đầy đủ của tất cả các file ảnh trong thư mục con vào danh sách
            self.y.extend([current_class]*len(filenames)) # Thêm các class_id tương ứng vào mỗi ảnh
            current_class += 1

        if one_hot:#120 class y[i]=3 -> [0,0,0,1, ...0]
            for _ in range(len(self.y)):
                tmp = [0]*self.num_class
                tmp[self.y[_]] = 1
                self.y[_] = tmp

        self.y = np.array(self.y,dtype = 'float32') #Chuyển danh sách label sang mảng numpy.
        self.y = torch.tensor(self.y,dtype= torch.float32) #Chuyển mảng numpy sang tensor PyTorch.

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return [self.X[idx],self.y[idx]]
    
    def get_splits(self,n_test=0.33):
        test_size = int(len(self.X)*n_test) # Tính kích thước của tập test dựa trên tỉ lệ n_test
        return random_split(self,[len(self.X)-test_size,test_size]) # Chia dataset thành 2 phần là train và test
    
if __name__=='__main__':
    dataset = DogDataset(Image_folder,one_hot=0)
    print(dataset.class_names)
    train_set, test_set = dataset.get_splits()
    print(train_set)
    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
