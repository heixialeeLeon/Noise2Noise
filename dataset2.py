import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import os
import skimage
import torchvision.transforms.functional as tvF
from transform.image_show import *
import random
from random import choice
import glob

def label_to_hongzhang(label_name):
    noise_name = ""
    if "tar1.bmp" in label_name:
        noise_name = label_name.replace("tar1.bmp","src1.bmp")
    elif "tar.bmp" in label_name:
        noise_name = label_name.replace("tar.bmp","src.bmp")
    return noise_name

def label_to_hongzhang2(label_name):
    file_name = label_name[:-4]
    noise_name = file_name+".jpg"
    return noise_name

def Transform(size):
    return Compose([
        Resize(size),
        ToTensor()
    ])

class HongZhang_Dataset2(Dataset):
    def __init__(self, root_folder, size=(256,256)):
        self.noise_dir = os.path.join(root_folder,"image")
        self.label_dir = os.path.join(root_folder,"label")
        self.files = glob.glob(self.label_dir+'/*.bmp')
        self.transform = Transform(size)
        self.label_ids =[]
        self.noise_ids = []
        for i in range(len(self.files)):
            _,fn = os.path.split(self.files[i])
            self.label_ids.append(fn)
            self.noise_ids.append(label_to_hongzhang(fn))
        print("label: {}, nosie: {}".format(len(self.label_ids),len(self.noise_ids)))
        # assert(len(self.label_dir) == len(self.noise_ids))

    def __getitem__(self,index):
        # print("label: {}".format(self.label_ids[index]))
        # print("noise: {}".format(self.noise_ids[index]))
        noise_image = Image.open(os.path.join(self.noise_dir, self.noise_ids[index])).convert('RGB')
        label_image = Image.open(os.path.join(self.label_dir, self.label_ids[index])).convert('RGB')
        return self.transform(label_image), self.transform(noise_image)

    def __len__(self):
        return len(self.label_ids)

class HongZhang_Dataset3(Dataset):
    def __init__(self, root_folder, size=(256,256)):
        self.noise_dir = os.path.join(root_folder,"src3")
        self.label_dir = os.path.join(root_folder,"dst3")
        self.files = glob.glob(self.label_dir+'/*.bmp')
        self.transform = Transform(size)
        self.label_ids =[]
        self.noise_ids = []
        for i in range(len(self.files)):
            _,fn = os.path.split(self.files[i])
            self.label_ids.append(fn)
            self.noise_ids.append(fn)
            # self.noise_ids.append(label_to_hongzhang2(fn))
        print("label: {}, nosie: {}".format(len(self.label_ids),len(self.noise_ids)))
        # assert(len(self.label_dir) == len(self.noise_ids))

    def __getitem__(self,index):
        # print("label: {}".format(self.label_ids[index]))
        # print("noise: {}".format(self.noise_ids[index]))
        noise_image = Image.open(os.path.join(self.noise_dir, self.noise_ids[index])).convert('RGB')
        label_image = Image.open(os.path.join(self.label_dir, self.label_ids[index])).convert('RGB')
        return self.transform(label_image), self.transform(noise_image)

    def __len__(self):
        return len(self.label_ids)

class HongZhang_TestDataset(Dataset):
    def __init__(self, root_folder, size=(256,256)):
        self.files = glob.glob(root_folder+'/*.bmp')
        self.transform = Transform(size)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    # dataset = HongZhang_Dataset2("/data_1/data/红章图片", (256, 256))
    # for item in dataset:
    #     print(item[0].shape)
    #     print(item[1].shape)
    #     CV2_showTensors(item[0],item[1])

    # """
    # for the data loader
    # """
    # dataset = HongZhang_Dataset2("/data_1/data/红章图片", (256, 256))
    # loader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)
    # for item in loader:
    #     CV2_showTensors(item[0], item[1],timeout=3000)

    # """
    # for the test dataset
    # """
    # dataset = HongZhang_TestDataset("/data_1/data/红章图片/test/hongzhang", (256, 256))
    # for item in dataset:
    #     CV2_showTensors(item)

    """
    for the dataset3
    """
    dataset = HongZhang_Dataset3("/data_1/data/红章图片/6_12", (256, 256))
    for item in dataset:
        print(item[0].shape)
        print(item[1].shape)
        CV2_showTensors(item[0],item[1])