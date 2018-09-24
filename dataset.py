import os, time
from PIL import Image
# import matplotlib.pyplot as plt
# import itertools
# import pickle
# import imageio
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

# dataset
class MyDataset(data.Dataset):
    def __init__(self, data_path, data_txt, transform=None, target_transform=None):
        file = open(data_txt, 'r')
        names = []
        for line in file:
            words = line.split()
            names.append((words[0], words[1]))
        self.names = names
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        value_name, target_name = self.names[index]
        value = Image.open(self.data_path+value_name).convert('RGB')
        target = Image.open(self.data_path+target_name).convert('RGB')
        if self.transform and self.target_transform is not None:
            value = self.transform(value)
            target = self.target_transform(target)
        return value, target

    def __len__(self):
        return len(self.names)

# parameters and data_loader
transform_0 = transforms.Compose([
    #transforms.Resize(1024),
    transforms.RandomCrop(32, 32),  # as paper says
    transforms.ToTensor(),
    ]
)

batch_size = 128

train_data = MyDataset('D:\\STU\\SRproject\\SR\\dataset',
                      'D:\\STU\\SRproject\\SR\\dataset\\imgs.txt',
                      transform=transform_0, target_transform=transform_0)
train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

# results save folder
if not os.path.isdir('Mydata_CinCGAN_results'):
    os.mkdir('Mydata_CinCGAN_results')
if not os.path.isdir('Mydata_CinCGAN_results/Random_results'):
    os.mkdir('Mydata_CinCGAN_results/Random_results')
if not os.path.isdir('Mydata_CinCGAN_results/Fixed_results'):
    os.mkdir('Mydata_CinCGAN_results/Fixed_results')
