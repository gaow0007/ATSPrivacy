import imp
import os, sys
from random import random, shuffle
sys.path.insert(0, './')
import torch
import numpy as np
import torchvision
import inversefed
from inversefed.data.data_processing import _build_cifar100, _get_meanstd, _build_imagenet, _build_celeba_gender, _build_celeba_identity, _build_celeba_mlabel, _build_celeba_smile, _build_celeba_face_align_mlabel
import torchvision.transforms as transforms
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import Classification, LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F

import policy
from datasets import load_dataset
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
import random
from pathlib import Path


policies = policy.policies


def create_model(opt):
    arch = opt.arch
    if opt.data == 'cifar100':
        model, _ = inversefed.construct_model(arch, num_classes=100, num_channels=3)
    elif opt.data == 'FashionMinist':
        model, _ = inversefed.construct_model(arch, num_classes=10, num_channels=1)
    elif opt.data == 'ImageNet':
        model, _ = inversefed.construct_model(arch, num_classes=25, num_channels=3)
    elif opt.data in ['CelebA_Gender', 'CelebA_Smile']: #Binary classification
        model, _ = inversefed.construct_model(arch, num_classes=2, num_channels=3)
    elif opt.data == 'CelebA_Identity': #Identity classification
        model, _ = inversefed.construct_model(arch, num_classes=500, num_channels=3)
    elif opt.data == 'CelebA_MLabel': #Multilabel classification
        model, _ = inversefed.construct_model(arch, num_classes=40, num_channels=3)
    elif opt.data == 'CelebAFaceAlign_MLabel': #Multilabel classification
        model, _ = inversefed.construct_model(arch, num_classes=40, num_channels=3)
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





def build_vit_transform(normalize=True, policy_list=list(), opt=None, defs=None, mean_std=None, scale_size=None):
    from torchvision.transforms import (CenterCrop, 
                                        Compose, 
                                        Normalize, 
                                        RandomHorizontalFlip,
                                        RandomResizedCrop, 
                                        Resize, 
                                        ToTensor)
    
    mean, std = mean_std
    normalize = Normalize(mean=mean, std=std)
    mode = opt.mode

    if mode !=  'crop':
        transform_list = list()

    elif mode == 'crop':
        transform_list = [RandomResizedCrop(scale_size),
                            transforms.RandomHorizontalFlip()]

    if len(policy_list) > 0 and mode == 'aug':

        transform_list = [RandomResizedCrop(scale_size),
                            transforms.RandomHorizontalFlip()]
        transform_list.append(construct_policy(policy_list))

    transform_list.extend([ToTensor(), normalize])
    _train_transforms = Compose(
            transform_list
        )

    _val_transforms = Compose(
            [
                Resize(scale_size),
                CenterCrop(scale_size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    return train_transforms, val_transforms


def build_transform(normalize=True, policy_list=list(), opt=None, defs=None):
    mode = opt.mode
    if opt.data == 'cifar100':
        data_mean, data_std = inversefed.consts.cifar10_mean, inversefed.consts.cifar10_std
    elif opt.data == 'FashionMinist':
        data_mean, data_std  = (0.1307,), (0.3081,)
    elif opt.data == 'ImageNet':
        #TODO use constant or recompute the mean and std ?
        data_mean, data_std = inversefed.consts.imagenet_mean, inversefed.consts.imagenet_std
    elif opt.data.startswith('CelebA'):
        data_mean, data_std = inversefed.consts.celeba_mean, inversefed.consts.celeba_std
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

    elif opt.data == 'ImageNet':
        transform_list = [transforms.Resize(256),
                            transforms.CenterCrop(224)]
        
        if len(policy_list) > 0 and mode == 'aug':
            transform_list.append(construct_policy(policy_list))

    elif opt.data.startswith('CelebA'):
        # transform_list = [transforms.Resize((128, 128))]
        transform_list = [transforms.Resize((112, 112))]
        
        if len(policy_list) > 0 and mode == 'aug':
            transform_list.append(construct_policy(policy_list))
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



def vit_preprocess(opt, defs, valid=False): 
    defs.validate = 1 # TODO: important configuration
    data_arrow = load_dataset(opt.data)
    train_ds = data_arrow['train']
    val_ds = data_arrow['test']
    label_key = 'fine_label' if opt.data=='cifar100' else 'label'
    id2label = {id:label for id, label in enumerate(train_ds.features[label_key].names)}
    label2id = {label:id for id,label in id2label.items()}
    # data transform 

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[label_key] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    if len(opt.aug_list) > 0:
        policy_list = split(opt.aug_list)
    else:
        policy_list = []
    home = Path.home().as_posix() 
    pretrain_path = 'google/vit-base-patch16-224-in21k' if not os.path.exists(home + '/.cache/huggingface/transformers/vit-base-patch16-224-in21k') else home + '/.cache/huggingface/transformers/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(pretrain_path)
    mean, std =feature_extractor.image_mean, feature_extractor.image_std
    scale_size = feature_extractor.size 

    train_transforms, val_transforms = build_vit_transform(True, policy_list, opt, defs, (mean,std), scale_size)
    if opt.tiny_data:
    # 10% data sample
        indices = [i for i in range(len(train_ds))]
        random.shuffle(indices)
        subset_indices = indices[:int(0.1*len(train_ds))] 
        train_ds = train_ds.select(subset_indices)

    train_ds.set_transform(train_transforms)
    if valid:
        val_ds.set_transform(train_transforms)
    else:
        val_ds.set_transform(val_transforms)
    trainloader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, batch_size=128, 
            shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    validloader = torch.utils.data.DataLoader(val_ds, collate_fn=collate_fn, batch_size=256,
            shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    loss_fn = Classification()
    model = ViTForImageClassification.from_pretrained(pretrain_path,
                                                  num_labels=len(id2label.keys()),
                                                  id2label=id2label,
                                                  label2id=label2id)
    
    return loss_fn, trainloader, validloader, model, (mean,std), scale_size



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

    elif opt.data == 'ImageNet':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('ImageNet', defs)
        trainset, validset = _build_imagenet('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=1
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=64,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

        return loss_fn, trainloader, validloader

    elif opt.data == 'CelebA_Gender':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebA_Gender', defs)
        trainset, validset = _build_celeba_gender('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=10
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

        return loss_fn, trainloader, validloader

    elif opt.data == 'CelebA_Identity':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebA_Identity', defs)
        trainset, validset = _build_celeba_identity('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=10
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

        return loss_fn, trainloader, validloader
    elif opt.data == 'CelebA_MLabel':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebA_MLabel', defs)
        trainset, validset = _build_celeba_mlabel('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=10
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

        return loss_fn, trainloader, validloader

        
    elif opt.data == 'CelebA_Smile':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebA_Smile', defs)
        trainset, validset = _build_celeba_smile('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=10
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

        return loss_fn, trainloader, validloader
    elif opt.data == 'CelebAFaceAlign_MLabel':

        loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CelebAFaceAlign_MLabel', defs)
        trainset, validset = _build_celeba_face_align_mlabel('~/data/')

        if len(opt.aug_list) > 0:
            policy_list = split(opt.aug_list)
        else:
            policy_list = []
        if not valid:
            trainset.transform = build_transform(True, policy_list, opt, defs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                    shuffle=True, drop_last=True, num_workers=24, pin_memory=True)
        if opt.tiny_data:
            print('Use tiny dataset')
            defs.validate=10
        # 10% data sample
            subset_indices = torch.randperm(len(trainset))[:int(0.1*len(trainset))]
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                        drop_last=False, num_workers=16, pin_memory=True, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))

        if valid:
            validset.transform = build_transform(True, policy_list, opt, defs)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128,
                shuffle=False, drop_last=True, num_workers=16, pin_memory=True)

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
                # max_iterations=100, #debug
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
    elif opt.optim == 'inversed_large':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                # max_iterations=100, #debug
                max_iterations=4800,
                total_variation=1e-1,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    else:
        raise NotImplementedError
    return config


