
import os
import torch
from params import *
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from albumentations import ElasticTransform

from torchvision.transforms.functional import adjust_gamma

PARAMS = Params()

def gamma_correction(image, gamma=2.2):
    return adjust_gamma(image, gamma=gamma)

elastic_transform_params = {
    'alpha': 50, 
    'sigma': 30,
    'alpha_affine': 70, 
    'p': 0.35, 
    'border_mode': 2,
}



paddy_labels = {'Cholangiocarcinoma':0, 'HCC':1, 'Normal_Liver':2}

class PaddyDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {'Cholangiocarcinoma':0, 'HCC':1, 'Normal_Liver':2}
        self.transform = transform
        self.albumentation_transform = ElasticTransform(**elastic_transform_params)
        self.data_info = self.get_img_info(data_dir)
        
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        img = ImageOps.equalize(img)
        img = gamma_correction(img, gamma=2.2)
        if self.transform == train_transform:
            img_np = np.array(img)
            augmented = self.albumentation_transform(image=img_np)
            img = Image.fromarray(augmented['image'])
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print(sub_dir)
                    label = paddy_labels[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


train_transform = transforms.Compose([
    transforms.Resize([PARAMS.img_size, PARAMS.img_size]),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.7,1.0),),
    # transforms.RandomRotation(10),
    # elastic_transform, 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5002, 0.5002, 0.5002], std=[0.2874, 0.2874, 0.2874]),
])

val_transform = transforms.Compose([
    transforms.Resize([PARAMS.img_size, PARAMS.img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5002, 0.5002, 0.5002], std=[0.2874, 0.2874, 0.2874]),
])

data = r"datasets/train"


datasets = PaddyDataSet(data_dir=data,transform=train_transform)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


