# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn import WideResNet
from tqdm import tqdm
from imagenet_downsampled import ImageNetDS
from load_corrupted_ours_hierarchical import CIFAR10, CIFAR100

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.5, help='The label corruption probability.')
parser.add_argument('--num_classes_pretrained_net', type=int, default=1000, choices=[1000, 365], help='1000 for ImageNet, 365 for Places365.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=256)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/cifar', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./snapshots/baseline', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

print(args)


state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

train_transform = trn.Compose([trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(),
                               trn.ToTensor(), trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


if args.dataset == 'cifar10':
    train_data = CIFAR10(
        args.data_path, True, False, 0, corruption_prob=args.corruption_prob, corruption_type='unif', transform=train_transform)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform)
    # # VAL SET
    # test_indices = np.random.choice(50000, 10000, replace=False)
    # train_indices = np.array(list(set(np.arange(50000)) - set(test_indices)))
    # test_data.test_data = np.copy(train_data.train_data[test_indices])
    # train_data.train_data = np.copy(train_data.train_data[train_indices])
    # #
    num_classes = 10
elif args.dataset == 'cifar100':
    train_data = CIFAR100(
        args.data_path, True, False, 0, corruption_prob=args.corruption_prob, corruption_type='unif', transform=train_transform)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform)
    num_classes = 100
    # # VAL SET
    # test_indices = np.random.choice(50000, 10000, replace=False)
    # train_indices = np.array(list(set(np.arange(50000)) - set(test_indices)))
    # test_data.test_data = np.copy(train_data.train_data[test_indices])
    # train_data.train_data = np.copy(train_data.train_data[train_indices])
    # #


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(args.num_classes_pretrained_net)
else:
    net = WideResNet(args.layers, args.num_classes_pretrained_net, args.widen_factor, dropRate=args.droprate)

net = nn.DataParallel(net)

model_found = False
# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, 'imagenet_' + args.model +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            model_found = True
            break
    if not model_found:
        assert False, "could not resume"

net = net.module

class FineTuneModel(nn.Module):
    def __init__(self, model):
        super(FineTuneModel, self).__init__()

        self.trunk = nn.Sequential(*list(model.children())[:-1])
        # self.classifier = list(model.children())[-1]

        self.classifier = nn.Linear(128, num_classes)
        self.unfreeze(True)

    def unfreeze(self, unfreeze):
        for p in self.trunk.parameters():
            p.requires_grad = unfreeze

    def forward(self, x):
        x = self.trunk(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        return self.classifier(x)   #, x


net = FineTuneModel(net)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# optimizer = torch.optim.SGD(
#     net.parameters(), state['learning_rate'], momentum=state['momentum'],
#     weight_decay=state['decay'], nesterov=True)

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                            state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////

def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for data, target in tqdm(train_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.95 + float(loss) * 0.05

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_' + args.model +
                            '_baseline_epoch_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, args.dataset + '_' + args.model +
                             '_baseline_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
