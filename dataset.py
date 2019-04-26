import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
import os
import skimage
import torchvision.transforms.functional as tvF
from transform.image_show import *
import random
from random import choice
from string import ascii_letters

def Gaussian_Noise(img, mean=0, var=0.01):
    img = np.array(img)
    img = skimage.util.random_noise(img, 'gaussian', mean=mean, var=var)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

class Training_Dataset(Dataset):
    def __init__(self, image_dir, image_size, noise_param ,crop=True):
        self.image_dir = image_dir
        self.image_list = os.listdir(self.image_dir)
        self.image_size = image_size
        self.noise = noise_param[0]
        self.noise_parameter = noise_param[1]
        self.image_crop = crop

    def _add_gaussian_noise(self, image):
        """
        Added only gaussian noise
        """
        w, h = image.size
        c = len(image.getbands())
        std = np.random.uniform(0, self.noise_parameter)
        _n = np.random.normal(0, std, (h, w, c))
        noise_image = np.array(image) + _n
        noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_image)

    def _add_m_bernoulli_noise(self, image):
        """
        Multiplicative bernoulli
        """
        sz = np.array(image).shape[0]
        prob_ = random.uniform(0, self.noise_parameter)
        mask = np.random.choice([0, 1], size=(sz, sz), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        noise_image = np.multiply(image, mask).astype(np.uint8)
        return Image.fromarray(noise_image)

    def _add_text_overlay(self, image):
        """
        Add text overlay to image
        """
        assert self.noise_parameter < 1, 'Text parameter should be probability of occupancy'

        w, h = image.size
        c = len(image.getbands())

        serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        # if platform == 'linux':
        #     serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        # else:
        #     serif = 'Times New Roman.ttf'

        text_img = image.copy()
        text_draw = ImageDraw.Draw(text_img)
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        max_occupancy = np.random.uniform(0, self.noise_parameter)

        def get_occupancy(x):
            y = np.array(x, np.uint8)
            return np.sum(y) / y.size

        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break
        return text_img


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.image_list[index])
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.image_size, Image.BILINEAR)
        noise_image = img
        if self.noise == 'Gaussain':
            noise_image = self._add_gaussian_noise(img)
        elif self.noise == 'Text':
            noise_image = self._add_text_overlay(img)
        elif self.noise == "Multiplicative_bernoulli":
            noise_image = self._add_m_bernoulli_noise(img)
        return tvF.to_tensor(img), tvF.to_tensor(noise_image)

"""
  for the dataset test 
"""
# if __name__ == "__main__":
#     dataset = Training_Dataset("/data_1/data/VOC2007/VOCdevkit/VOC2007/JPEGImages",(256,256,),"Gaussain")
#     for item in dataset:
#         #PIL_ShowTensor2(item[0],item[1])
#         #PIL_ShowTensor(item[0])
#         #PIL_ShowTensor3(item[0],item[1],item[1])
#         #PIL_ShowTensor_Timeout(item[0],3000)
#         img1=tvF.to_pil_image(item[0])
#         img2=tvF.to_pil_image(item[1])
#         #CV2_showPILImage2(img1,img2,1000)
#         CV2_showPILImages(img1,img2)

"""
  for the dataloader test 
"""
if __name__ == "__main__":
    dataset = Training_Dataset("/data_1/data/Noise2Noise/train/291", (300, 300), ("Multiplicative_bernoulli",0.7))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    # for item in train_loader:
    #     print(item[0].shape)
    #     print(item[1].shape)
    for batch_idx, (origin, noise) in enumerate(train_loader):
        # print(batch_idx)
        # print(source.shape)
        # print(target.shape)
        CV2_showTensors(noise,origin)