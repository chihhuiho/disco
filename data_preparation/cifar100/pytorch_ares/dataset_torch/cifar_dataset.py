from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def cifar10(batchsize, cifar10_path): 
    
    
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory= False, drop_last= False)
    test_loader.name = "cifar10"
    test_loader.batch = batchsize
    return test_loader

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def cifar10_dist(batchsize, cifar10_path, need_dataset=False): 
    
    
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = CIFAR10(root=cifar10_path, train=False, download=False, transform=transform)
    test_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory= False, drop_last= False)
    test_loader.name = "cifar10"
    test_loader.batch = batchsize
    if need_dataset:
        return cifar
    else:
        return test_loader
