import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os
import torchvision.transforms.functional as tvF
from model.unet import UNet
from model.srresnet import SRResnet
from model.eesp.eesp_segmentation import EESPNet_Seg
import time
from dataset import Training_Dataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description="train noise2noise model")

# data loader parameters
parser.add_argument("--image_dir", type=str, required=False, default="/data_1/data/Noise2Noise/coco_data/train/",help="train image dir")
parser.add_argument("--test_dir", type=str, required=False, default="/data_1/data/Noise2Noise/coco_data/test",help="test image dir")

# training parameters
parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--batch_size", type=int, default=16,help="batch size")
parser.add_argument("--epochs", type=int, default=60,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=5,help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--steps_show", type=int, default=10,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoints",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda:0",help="device description")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--resume_model", type=str, default=None, help="resume model path")

# model parameter
parser.add_argument("--model", choices=['unet','srresnet','eesp'],type=str, default="unet", help="which model to train")

# Loss parameters
parser.add_argument("--loss", choices=['l1','l2'],type=str, default="l2", help="loss type")

# Noise parameters
parser.add_argument("--noise", choices=['Gaussain','Text','Multiplicative_bernoulli'],type=str, default="Gaussain", help="noise type")
parser.add_argument("--noise_param", type=float, default=50, help="noise parameter")

args = parser.parse_args()
print(args)

def save_model(model,epoch):
    '''save model for eval'''

    ckpt_name = '/denoise_epoch_{}.pth'.format(epoch)
    path = args.output_path
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def resume_model(model, model_path):
    print("Resume model from {}".format(args.resume_model))
    model.load_state_dict(torch.load(model_path))

def train():
    # prepare the dataloader
    device = torch.device(args.devices if torch.cuda.is_available() else "cpu")
    dataset = Training_Dataset(args.image_dir, (args.image_size, args.image_size), (args.noise,args.noise_param))
    dataset_length = len(dataset)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # choose the model
    if args.model == "unet":
        model = UNet(in_channels=args.image_channels, out_channels=args.image_channels)
    elif args.model == "srresnet":
        model = SRResnet(args.image_channels, args.image_channels)
    elif args.model == "eesp":
        model = EESPNet_Seg(args.image_channels,2)
    else:
        model = UNet(in_channels=args.image_channels, out_channels=args.image_channels)
    model = model.to(device)

    # choose the loss type
    if args.loss == "l2":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()

    # resume the mode if needed
    if args.resume_model:
        resume_model(model, args.resume_model)

    optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
    #scheduler = lr_scheduler.StepLR(optim, step_size=args.scheduler_step, gamma=0.5)
    scheduler = lr_scheduler.MultiStepLR(optim,milestones=[20,30,40],gamma=0.1)
    model.train()
    print(model)

    # start to train
    print("Starting Training Loop...")
    since = time.time()
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        scheduler.step()
        for batch_idx, (target, source) in enumerate(train_loader):
            source = source.to(device)
            target = target.to(device)
            denoised_source = model(source)
            loss = criterion(denoised_source, Variable(target))
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item() * source.size(0)
            if batch_idx % args.steps_show == 0:
                print('{}/{} Current loss {}'.format(batch_idx,len(train_loader),loss.item()))
        epoch_loss = running_loss / dataset_length
        print('{} Loss: {:.4f}'.format('current ' + str(epoch), epoch_loss))
        if (epoch + 1) % args.save_per_epoch == 0:
            save_model(model, epoch + 1)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    train()