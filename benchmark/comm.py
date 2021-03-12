import os, sys
sys.path.insert(0, './')
import torch
import numpy as np
import torchvision
import inversefed
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import policy
policies = policy.policies


def create_model(opt):
    arch = opt.arch
    if opt.data == 'cifar100':
        model, _ = inversefed.construct_model(arch, num_classes=100, num_channels=3)
    elif opt.data == 'FashionMinist':
        model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=1)
    return model


class sub_transform:
    def __init__(self, policy_list):
        self.policy_list = policy_list


    def __call__(self, img):
        idx = np.random.randint(0, len(self.policy_list))
        select_policy = self.policy_list[idx]
        for policy_id in select_policy:
            img = policies[policy_id](img)
        return img


def construct_policy(policy_list):
    if isinstance(policy_list[0], list):
        return sub_transform(policy_list)
    elif isinstance(policy_list[0], int):
        return sub_transform([policy_list])
    else:
        raise NotImplementedError


def build_transform(normalize=True, policy_list=list(), opt=None, defs=None):
    mode = opt.mode
    if opt.data == 'cifar100':
        data_mean, data_std = inversefed.consts.cifar10_mean, inversefed.consts.cifar10_std
    elif opt.data == 'FashionMinist':
        data_mean, data_std  = (0.1307,), (0.3081,)
    else:
        raise NotImplementedError

    # data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if mode !=  'crop':
        transform_list = list()

    elif mode == 'crop':
        transform_list = [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]

    if len(policy_list) > 0 and mode == 'aug':

        transform_list = [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]
        transform_list.append(construct_policy(policy_list))


    if opt.data == 'FashionMinist':
        transform_list = [lambda x: transforms.functional.to_grayscale(x, num_output_channels=3)] + transform_list
        transform_list.append(lambda x: transforms.functional.to_grayscale(x, num_output_channels=1))
        transform_list.append(transforms.Resize(32))


    print(transform_list)


    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x),
    ])

    transform = transforms.Compose(transform_list)
    return transform


def split(aug_list):
    if '+' not in aug_list:
        return [int(idx) for idx in aug_list.split('-')]
    else:
        ret_list = list()
        for aug in aug_list.split('+'):
            ret_list.append([int(idx) for idx in aug.split('-')])
        return ret_list

def preprocess(opt, defs, valid=False):
    if opt.data == 'cifar100':
        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR100', defs)
        trainset, validset = _build_cifar100('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
                    shuffle=True, drop_last=False, num_workers=4, pin_memory=True)


        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=defs.batch_size,
                shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        return loss_fn, trainloader, validloader

    elif opt.data == 'FashionMinist' :
        loss_fn, _, _ =  inversefed.construct_dataloaders('CIFAR100', defs)
        trainset = torchvision.datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        validset = torchvision.datasets.FashionMNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        # key
        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list) # [int(idx) for idx in opt.aug_list.split('-')]
        else:
            policy_list = []
        tlist = policy_list if not valid else list()
        trainset.transform = build_transform(True, tlist, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
                    shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

        tlist = list() if not valid else policy_list
        validset.transform = build_transform(True, tlist, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=defs.batch_size,
                shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        return loss_fn, trainloader, validloader
    else:
        raise NotImplementedError



def create_config(opt):
    print(opt.optim)
    if opt.optim == 'inversed':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif opt.optim == 'inversed-zero':
        config = dict(signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif opt.optim == 'inversed-sim-out':
        config = dict(signed=True,
            boxed=True,
            cost_fn='out_sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif opt.optim == 'inversed-sgd-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='sgd',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif opt.optim == 'inversed-LBFGS-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=1e-4,
                optim='LBFGS',
                restarts=16,
                max_iterations=300,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=False,
                scoring_choice='loss')
    elif opt.optim == 'inversed-adam-L1':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l1',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif opt.optim == 'inversed-adam-L2':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l2',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif opt.optim == 'zhu':
        config = dict(signed=False,
                        boxed=False,
                        cost_fn='l2',
                        indices='def',
                        weights='equal',
                        lr=1e-4,
                        optim='LBFGS',
                        restarts=2,
                        max_iterations=50, # ??
                        total_variation=1e-3,
                        init='randn',
                        filter='none',
                        lr_decay=False,
                        scoring_choice='loss')
        seed=1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        import random
        random.seed(seed)
    else:
        raise NotImplementedError
    return config


