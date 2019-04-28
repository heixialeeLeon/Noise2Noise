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
import math

def Gaussian_Noise(img, mean=0, var=0.01):
    img = np.array(img)
    img = skimage.util.random_noise(img, 'gaussian', mean=mean, var=var)
    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

class HongZhang_Dataset(Dataset):
    def __init__(self, image_dir, redstamp_dir):
        self.image_dir = image_dir
        self.redstamp_dir = redstamp_dir
        self.image_list = os.listdir(self.image_dir)
        self.redstamp_list = os.listdir(self.redstamp_dir)
        self.target_w = 512
        self.target_h = 512
        self.red_stamp_w = 120
        self.red_stamp_h = 120
        self.margin = 100

    def _get_redstamp(self):
        index = random.randint(0,len(self.redstamp_list)-1)
        redstamp_file = os.path.join(self.redstamp_dir,self.redstamp_list[index])
        red_stamp = cv2.imread(redstamp_file)
        red_stamp_rotation = self._rotate_bound(red_stamp, random.randint(0, 360))
        red_stamp_rotation =  cv2.resize(red_stamp_rotation,(self.red_stamp_h,self.red_stamp_w),interpolation=cv2.INTER_CUBIC)
        red_stamp_rotation = self._color_change(red_stamp_rotation)
        return red_stamp_rotation

    def _rotate_bound(self, image, angle):
        height, width = image.shape[:2]
        heightNew = int(width * abs(math.sin(math.radians(angle))) + height * abs(math.cos(math.radians(angle))))
        widthNew = int(height * abs(math.sin(math.radians(angle))) + width * abs(math.cos(math.radians(angle))))
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        return cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    def _color_change(self,img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        w = img_hsv.shape[1]
        h = img_hsv.shape[0]
        x_ratio = np.random.uniform(0.95, 1.02)
        y_ratio = np.random.uniform(0.55, 1)
        z_ratio = np.random.uniform(0.6, 1)
        # print(z_ratio)
        for i in range(0, h):
            for j in range(0, w):
                if int(img_gray[i, j]) < 170:
                    # print(1)
                    x = int(x_ratio * float(img_hsv[i, j, 0]))
                    img_hsv[i, j, 0] = x
                    y = int(y_ratio * float(img_hsv[i, j, 1]))
                    img_hsv[i, j, 1] = y
                    z = int(z_ratio * float(img_hsv[i, j, 2]))
                    img_hsv[i, j, 2] = z
        img_change = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_change

    def _crop_image(self,image):
        image_shape = image.shape
        w = image_shape[1]
        h = image_shape[0]
        crop_x = random.randint(20, int(w - self.target_w - self.margin))
        crop_y = random.randint(20, int(h - self.target_h - self.margin))
        img = image[crop_y:crop_y + self.target_h, crop_x:crop_x + self.target_w]
        return img


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.image_list[index])
        img = cv2.imread(image_path)
        img = self._crop_image(img)
        noise_img = img.copy()
        redstamp = self._get_redstamp()

        # generate mask
        red_stamp_gray = cv2.cvtColor(redstamp, cv2.COLOR_BGR2GRAY)
        red, mask = cv2.threshold(red_stamp_gray, 155, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # roi locate
        roi_x = random.randint(0, self.target_w - self.red_stamp_w - 1)
        roi_y = random.randint(0, self.target_h - self.red_stamp_h - 1)
        roi = noise_img[roi_y:roi_y + self.red_stamp_h, roi_x:roi_x + self.red_stamp_w]

        # background and frontground
        bg = cv2.bitwise_and(roi, roi, mask=mask)
        fg = cv2.bitwise_and(redstamp, redstamp, mask=mask_inv)
        dst_roi = bg + fg

        # weighted the image
        weight_rate = np.random.uniform(0.7,0.95)
        dst_roi = cv2.addWeighted(dst_roi, weight_rate, roi, 1 - weight_rate, 0)

        noise_img[roi_y:roi_y + self.red_stamp_h, roi_x:roi_x + self.red_stamp_w] = dst_roi
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
        return tvF.to_tensor(img), tvF.to_tensor(noise_img)

"""
  for the dataset test 
"""
# if __name__ == "__main__":
#     dataset = Training_Dataset("/data_1/data/Noise2Noise/shenqingbiao/0202","/data_1/data/Noise2Noise/hongzhang")
#     for item in dataset:
#         #PIL_ShowTensor2(item[0],item[1])
#         #PIL_ShowTensor(item[0])
#         #PIL_ShowTensor3(item[0],item[1],item[1])
#         #PIL_ShowTensor_Timeout(item[0],3000)
#         #img1=tvF.to_pil_image(item[0])
#         #img2=tvF.to_pil_image(item[1])
#         #CV2_showPILImage2(img1,img2,1000)
#         #CV2_showPILImages(img1,img2)
#         CV2_showTensors(item[0],item[1])

"""
  for the dataloader test 
"""
if __name__ == "__main__":
    dataset = HongZhang_Dataset("/data_1/data/Noise2Noise/shenqingbiao/0202", "/data_1/data/Noise2Noise/hongzhang")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    # for item in train_loader:
    #     print(item[0].shape)
    #     print(item[1].shape)
    for batch_idx, (origin, noise) in enumerate(train_loader):
        # print(batch_idx)
        # print(source.shape)
        # print(target.shape)
        CV2_showTensors(noise,origin)