import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
import torchattacks
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchattacks import *
from utils import imsave
import argparse

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="pgd", choices=['fgsm', 'pgd', "bim"], help='attack_type')
parser.add_argument('--classifier', type=str, default="resnet18", help='pretrained classifier')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--skip_clean', action="store_true", default=False, help='debug mode')

args = parser.parse_args()



class imagenet_class(Dataset):
    def __init__(self, root, transform, label2idx, sample):
        self.img_path, self.label = [], []
        for label in os.listdir(root):
            for img in os.listdir(root + "/" + label)[:sample]:
                self.img_path.append(os.path.join(root,label, img))
                self.label.append(label2idx[label])
        self.transform = transform
        
    def __len__(self,):
        return len(self.img_path)
    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        
        img = self.transform(img)
        return img, self.label[idx]
    
class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

 
def main():   
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    ])



    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    class_idx = json.load(open("imagenet_class_index.json"))
    label2idx = {class_idx[str(k)][0]:k for k in range(len(class_idx))}

    imagenet_dataset = {}
    imagenet_dataset["train"] = imagenet_class(root=os.getcwd() + "/imagenet_train", transform=transform, label2idx=label2idx, sample=50)
    imagenet_dataset["val"] = imagenet_class(root=os.getcwd() + "/imagenet_val", transform=transform, label2idx=label2idx, sample=5)
    data_loader = {}
    data_loader["train"] = torch.utils.data.DataLoader(imagenet_dataset["train"], batch_size=32, shuffle=False)
    data_loader["val"] = torch.utils.data.DataLoader(imagenet_dataset["val"], batch_size=32, shuffle=False)

   
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.classifier == "resnet18":
        classifier = models.resnet18(pretrained=True)
    elif args.classifier == "resnet50":
        classifier = models.resnet50(pretrained=True)
    elif args.classifier == "alexnet":
        classifier = models.alexnet(pretrained=True)
    elif args.classifier == "vgg16":
        classifier = models.vgg16(pretrained=True)
    elif args.classifier == "mobilenet_v2":
        classifier = models.mobilenet_v2(pretrained=True)

    
    model = nn.Sequential(
        norm_layer,
        classifier
    ).to(device)

    model = model.eval()

    save_dir = "dataset"

    if args.attack == "fgsm":
        atk = FGSM(model, eps=8/255)
    elif args.attack == "pgd":
        atk = PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True)
    elif args.attack == "bim":
        atk = BIM(model, eps=8/255, alpha=2/255, steps=100)


    atk_method = args.attack

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, atk_method)):
        os.mkdir(os.path.join(save_dir, atk_method))
    if not os.path.exists(os.path.join(save_dir, atk_method, "train")):
        os.mkdir(os.path.join(save_dir, atk_method, "train"))
        os.mkdir(os.path.join(save_dir, atk_method, "train/clean"))
        os.mkdir(os.path.join(save_dir, atk_method, "train/adv"))
    if not os.path.exists(os.path.join(save_dir, atk_method, "val")):
        os.mkdir(os.path.join(save_dir, atk_method, "val"))
        os.mkdir(os.path.join(save_dir, atk_method, "val/clean"))
        os.mkdir(os.path.join(save_dir, atk_method, "val/adv"))

    # create training dataset
    cnt = 0
    for idx, (img, label) in tqdm(enumerate(data_loader["train"])):
        adv_images = atk(img, label)
 
        for i in range(img.shape[0]):
            if not args.skip_clean:
                imsave(img[i].cpu().data, os.path.join(save_dir, atk_method, "train/clean", str(cnt)))
            imsave(adv_images[i].cpu().data, os.path.join(save_dir, atk_method, "train/adv", str(cnt)))
            cnt += 1

            if args.debug:
                break
        if args.debug:
            break



    # create training dataset
    cnt = 0
    for idx, (img, label) in tqdm(enumerate(data_loader["val"])):
        adv_images = atk(img, label)
 
        for i in range(img.shape[0]):
            if not args.skip_clean:
                imsave(img[i].cpu().data, os.path.join(save_dir, atk_method, "val/clean", str(cnt)))
            imsave(adv_images[i].cpu().data, os.path.join(save_dir, atk_method, "val/adv", str(cnt)))
            cnt += 1

            if args.debug:
                break
        if args.debug:
            break

if __name__ == "__main__":
    main()
