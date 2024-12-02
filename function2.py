import os
import numpy as np
from torch.utils.data import Dataset, random_split
import torch
Image_folder = '/images/Images/'
class DogDataset(Dataset):
    """
    X = [paths to images]
    y = [class ids]
    class_names = [class name]
    """
    def __init__(self,path,*,num_class=120, transform=None):#path = ".../Images/"
        self.class_names = []
        self.num_class = num_class
        self.X = []
        self.y = []
        self.transform = transform
        current_class = 0
        for dirname,_, filenames in sorted(list(os.walk(path))[1:])[:num_class]: # os.walk will not follow any order
            class_name = dirname.split('/')[-1].split('-')[-1]
            self.class_names.append(class_name)
            self.X.extend(os.path.join(dirname,filename)for filename in sorted(filenames))
            self.y.extend([current_class]*len(filenames))
            current_class += 1
        self.y = np.array(self.y,dtype = 'int64') 
        self.y = torch.tensor(self.y,dtype= torch.int64) #CrossEntropyLoss require labels in type long
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return [self.X[index],self.y[index]]
    def get_splits(self, test_ratio=0.1, eval_ratio=0.2):
        torch.manual_seed(0)
        total_size = len(self.X)
        test_size = int(total_size * test_ratio)
        eval_size = int(total_size * eval_ratio)
        train_size = total_size - test_size - eval_size
        return random_split(self, [train_size, eval_size, test_size])