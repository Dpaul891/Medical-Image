import os
from random import random
from PIL import Image
from matplotlib import image
from numpy import imag
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        My Dataset for CAD with BUSI dataset
            param data_dir: str, path for the dataset
            param train: whether this is defined for training or testing
            param transform: torch.transform，data pre-processing pipeline
        """
        ### Begin your code ###
        self.data_dir = data_dir
        self.transform = transform
        self.dataset_type = 'train' if train else 'val'
        self.images, self.labels = self.get_data(self.data_dir)
        ### End your code ###
        
    
    def __getitem__(self, index): 
        '''
        Get sample-label pair according to index
        '''
        ### Begin your code ###
        image_path = self.images[index]
        img = Image.open(image_path).convert("RGB")
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        
        sample = {'image':img, 'label':label}
        return sample
        ### End your code ###

    def __len__(self): 
        '''return the size of the dataset'''
        ### Begin your code ###
        return len(self.images)
        ### End your code ###
        

    def get_data(self, data_dir):
        '''
        Load the dataset and store it in your own data structure(s)
        '''
        ### Begin your code ###
        if self.dataset_type == 'train':
            paths = open('./train_img_label.txt','r')
            ls = paths.readlines()
            ls = [x.strip() for x in ls]
            images = [x.split()[0] for x in ls]
            labels = [int(x.split()[1]) for x in ls]
        else:
            paths = open('./val_img_label.txt','r')
            ls = paths.readlines()
            ls = [x.strip() for x in ls]
            images = [x.split()[0] for x in ls]
            labels = [int(x.split()[1]) for x in ls]
        return images, labels
        ### End your code ###

if __name__ == '__main__':
    test_dataset = MyDataset(data_dir='COVID-19_Dataset', train=True, transform=None)
    print(test_dataset.labels[:10])
    print(test_dataset.__getitem__(10))

    
