import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os
import torchvision.transforms.functional as tvF
from model.unet import UNet
from model.srresnet import SRResnet
import time
from dataset import Training_Dataset
from torch.utils.data import Dataset, DataLoader
# from noise2noise_leon.Config import Config as conf
from transform.image_show import *

import argparse

parser = argparse.ArgumentParser(description="train noise2noise model")
# data loader parameters
parser.add_argument("--test_dir", type=str, required=False, default="/data_1/data/Noise2Noise/coco_data/test", help="test image dir")

# test parameters
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--output_path", type=str, default="../checkpoints",help="checkpoint dir")
parser.add_argument("--model_name", type=str, default="denoise_epoch_25.pth",help="load model name")
parser.add_argument("--denoised_dir", type=str, default="output",help="load model name")

# model parameters
parser.add_argument("--model", choices=['unet','srresnet'],type=str, default="unet", help="which model to train")

# Noise parameters
parser.add_argument("--noise", choices=['Gaussain','Text','Multiplicative_bernoulli'],type=str, default="Gaussain", help="noise type")
parser.add_argument("--noise_param", type=float, default=50, help="noise parameter")
args = parser.parse_args()
print(args)


def transpose(img,cpu=True):
    if not cpu:
        npimg = torch.squeeze(img).data.cpu().numpy()
    else:
        npimg = torch.squeeze(img).numpy()
    return np.transpose(npimg,(1,2,0))

def test():
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    test_dataset = Training_Dataset(args.test_dir, (args.image_size,args.image_size),(args.noise, args.noise_param))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model_path = os.path.join(args.output_path, args.model_name)
    print('Loading model from: {}'.format(model_path))

    # choose the model
    if args.model == "unet":
        model = UNet(in_channels=args.image_channels, out_channels=args.image_channels)
    elif args.model == "srresnet":
        model = SRResnet(args.image_channels, args.image_channels)
    else:
        model = UNet(in_channels=args.image_channels, out_channels=args.image_channels)
    print('loading model')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    result_dir = args.denoised_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for batch_idx, (target, source) in enumerate(test_loader):
        #PIL_ShowTensor(torch.squeeze(source))
        #PIL_ShowTensor2(torch.squeeze(source),torch.squeeze(noise))
        source = source.to(device)
        denoised_img = model(source).detach().cpu()
        CV2_showTensors(source.cpu(), target, denoised_img,timeout=3000)

if __name__ == "__main__":
    test()