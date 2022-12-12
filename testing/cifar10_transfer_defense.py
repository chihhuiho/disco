from robustbench.utils import load_model, clean_accuracy
import torch
import torch.nn as nn
import os
from robustbench.model_zoo.defense import inr
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse
import json
import numpy as np
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from advertorch.bpda import BPDAWrapper
import torchattacks
from torchattacks import *
from robustbench.model_zoo.defense import bit_depth_reduction
from robustbench.model_zoo.defense import jpeg_compression 
from robustbench.model_zoo.defense import randomization
from robustbench.model_zoo.defense import inr
from robustbench.model_zoo.defense import autoencoder
from robustbench.model_zoo.defense import stl
import torch.nn as nn
from advertorch.defenses import MedianSmoothing2D
from attacker.iterative_gradient_attack import FGM_L2
import foolbox as fb
from advertorch.attacks import L2BasicIterativeAttack



parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, help='attack')
parser.add_argument('--input_defense', type=str, default="disco", help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--stl_npy_name', type=str, default="64_p8_lm0.1", help='path to the stl numpy file')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=100, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--recursive_num', type=int, default=1, help='recursive_num for disco')

args = parser.parse_args()
assert args.trial is not None

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}



def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    
    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=4)
    
    return test_loader
 


def main():

    if args.debug:
        root = "log/defense_transfer/debug"
    else:
        root = "log/defense_transfer" 


    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, args.dataset)):
        os.mkdir(os.path.join(root, args.dataset))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root,  args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root,  args.dataset, args.input_defense, args.model_name)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack))


    filename = os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack, "result_" + args.trial + ".txt") 
 

    f = open(filename, "w")
    json.dump(args.__dict__, f, indent=2)
    f.write("\n")

    device = torch.device('cuda')
    batch_size=args.batch_size
    if args.debug:
        n_examples=50
    else:
        n_examples=args.batch_size
    
    model = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf').to(device)

    if args.attack == "autoattack":
        adversary = AutoAttack(model, norm='Linf', eps=8/255)
    elif args.attack == "FGSM":
        atk = FGSM(model, eps=8/255)
    elif args.attack == "FGSM_L2":
        '''
        atk = FGM_L2(model, loss_fn=nn.CrossEntropyLoss(),
				  mean=[0,0,0], std=[1,1,1], 
				  max_norm=4, # L2 norm bound
				  random_init=True)
        '''
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        atk = fb.attacks.L2FastGradientAttack()

    elif args.attack == "BIM_L2":
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        atk = fb.attacks.L2BasicIterativeAttack()
        #atk = L2BasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=4/255)

    elif args.attack == "CW":
        atk = CW(model, c=1, lr=0.01, steps=10, kappa=0)
    elif args.attack == "PGD":
        atk = PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True)
    elif args.attack == "PGDL2":
        atk = PGDL2(model, eps=8/255, alpha=0.2, steps=10)
    elif args.attack == "deepfool":	
        atk = DeepFool(model, steps=10)
    elif args.attack == "BIM":
        atk = BIM(model, eps=8/255, alpha=2/255, steps=100)
    elif args.attack == "RFGSM":
        atk = RFGSM(model, eps=8/255, alpha=2/255, steps=100)
    elif args.attack == "eotpgd":	
        atk = torchattacks.EOTPGD(model, eps=8/255, alpha=8/255, steps=40, eot_iter=10)
    elif args.attack == "tpgd":
        atk = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7)
    elif args.attack == "ffgsm":
        atk = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
    elif args.attack == "mifgsm":
        atk = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
    elif args.attack == "apgd":
        atk = torchattacks.APGD(model, eps=8/255)
    elif args.attack == "apgdt":
        atk = torchattacks.APGDT(model, eps=8/255)
    elif args.attack == "fab":
        atk = torchattacks.FAB(model)
    elif args.attack == "square":
        atk = torchattacks.Square(model)
    elif args.attack == "onepixel":
        atk = torchattacks.OnePixel(model)
    elif args.attack == "sparsefool":
        atk = torchattacks.SparseFool(model)
    elif args.attack == "di2fgsm":
        atk = torchattacks.DIFGSM(model, eps=8/255)
    elif args.attack == "tifgsm":
        atk = torchattacks.TIFGSM(model, eps=8/255)
    elif args.attack == "jitter":
        atk = torchattacks.Jitter(model, eps=8/255)


    if args.input_defense == "disco":
        defense = inr.INR(device, args.disco_path, height=32, width=32)
    elif args.input_defense == "liif":
        defense = inr.INR(device, args.liif_path, height=32, width=32)
    elif args.input_defense == "bit_depth_reduction":
        defense = bit_depth_reduction.BitDepthReduction(device)
    elif args.input_defense == "jpeg":
        defense = jpeg_compression.Jpeg_compresssion(device)
    elif args.input_defense == "randomization":
        defense = randomization.Randomization(device)
    elif args.input_defense == "autoencoder":
        defense = autoencoder.AutoEncoder("cifar10", args.autoencoder_path, device)
    elif args.input_defense == "stl":
        defense = stl.STL(device, "cifar", args.stl_npy_name)
    elif args.input_defense == "medianfilter":
        defense = nn.Sequential(MedianSmoothing2D(kernel_size=3))


    clean_acc_lst = []
    robust_acc_lst = []
    for i in range(args.repeat):
        print("==============")
        f.write("==============\n")
        print(str(i) + " Evaluation")
        f.write(str(i) + " Evaluation\n")


        clean_acc_all, robust_acc_all = 0, 0
        test_loader = load_cifar10(n_examples)
        iteration = int(10000/n_examples) if not args.debug else 1
        for idx, (x_clean, y_clean) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
 
            if args.attack == "autoattack":
                x_adv = adversary.run_standard_evaluation(x_clean, y_clean)
            elif args.attack == "FGSM_L2":
                #x_adv = atk.attack(x_clean, y_clean)
                _, x_adv, _ = atk(fmodel, x_clean, y_clean, epsilons=8/255)
            elif args.attack == "BIM_L2":
                #x_adv = atk.perturb(x_clean, y_clean)
                _, x_adv, _ = atk(fmodel, x_clean, y_clean, epsilons=8/255)
            else:
                x_adv = atk(x_clean, y_clean)

            if args.input_defense != "no_input_defense":
                x_clean = defense.forward(x_clean)
                x_adv = defense.forward(x_adv)


            clean_acc = clean_accuracy(model, x_clean, y_clean, batch_size=batch_size, device=device)
   
            clean_acc_all = clean_acc_all + clean_acc*n_examples/100.0
            robust_acc = clean_accuracy(model, x_adv, y_clean, batch_size=batch_size, device=device)
            robust_acc_all = robust_acc_all + robust_acc*n_examples/100.0

            if args.debug:
                break


        total = n_examples*iteration

        clean_acc_all = clean_acc_all/total*100.0
        print(f'Clean accuracy: {clean_acc_all:.4%}')
        f.write(f'Clean accuracy: {clean_acc_all:.4%}\n')
        clean_acc_lst.append(clean_acc_all)

        robust_acc_all = robust_acc_all/total*100.0
        print(f'Robust accuracy: {robust_acc_all:.4%}')
        f.write(f'Robust accuracy: {robust_acc_all:.4%}\n')
        robust_acc_lst.append(robust_acc_all)
        print("==============")
        f.write("==============\n")


    clean_acc_npy = np.array(clean_acc_lst)
    clean_acc_avg = np.mean(clean_acc_npy)
    clean_acc_std = np.std(clean_acc_npy)
     

    robust_acc_npy = np.array(robust_acc_lst)
    robust_acc_avg = np.mean(robust_acc_npy)
    robust_acc_std = np.std(robust_acc_npy)

    print(f'Avg clean accuracy: {clean_acc_avg:.4%}')
    f.write(f'Avg clean accuracy: {clean_acc_avg:.4%}\n')
    print(f'Std clean accuracy: {clean_acc_std:.4%}')
    f.write(f'Std clean accuracy: {clean_acc_std:.4%}\n')

    print(f'Avg robust accuracy: {robust_acc_avg:.4%}')
    f.write(f'Avg robust accuracy: {robust_acc_avg:.4%}\n')
    print(f'Std robust accuracy: {robust_acc_std:.4%}')
    f.write(f'Std robust accuracy: {robust_acc_std:.4%}\n')
 
    f.close()

if __name__ == "__main__":
    main()

