import os, sys
sys.path.insert(0, './')
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
import torch.nn.functional as F
from benchmark.comm import create_model, build_transform, preprocess, create_config
import policy
import copy

policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
opt = parser.parse_args()


# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']
num_images = 1



def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def get_batch_jacobian(net, x, target):
    net.eval()
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()

def calculate_dw(model, inputs, labels, loss_fn):
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError



def accuracy_metric(idx_list, model, loss_fn, trainloader, validloader):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []
    for idx in idx_list:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    jacobs, labels= get_batch_jacobian(model, ground_truth, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs, labels)



def reconstruct(idx, model, loss_fn, trainloader, validloader):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError
    
    # prepare data
    ground_truth, labels = [], []
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    # calcuate ori dW
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())

    metric = 'cos'

    # attack model
    model.eval()
    dw_list = list()
    dx_list = list()
    bin_num = 20
    noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds
    for dis_iter in range(bin_num+1):
        model.zero_grad()
        fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth + 1. / bin_num * (bin_num - dis_iter) * noise_input).detach()
        fake_dw = calculate_dw(model, fake_ground_truth, labels, loss_fn)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric=metric) for dw_a, dw_b in zip(fake_dw, input_gradient)]) / len(input_gradient)

        dw_list.append(dw_loss)

    interval_distance = cal_dis(noise_input, ground_truth, metric='L1') / bin_num


    def area_ratio(y_list, inter):
        area = 0
        max_area = inter * bin_num
        for idx in range(1, len(y_list)):
            prev = y_list[idx-1]
            cur = y_list[idx]
            area += (prev + cur) * inter / 2
        return area / max_area

    return area_ratio(dw_list, interval_distance)



def main():
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    old_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(opt.data, opt.arch, opt.epochs)))

    model.eval()
    metric_list = list()

    import time
    start = time.time()

    if True:
        sample_list = [200+i*5 for i in range(100)]
        metric_list = list()
        for attack_id, idx in enumerate(sample_list):
            metric = reconstruct(idx, model, loss_fn, trainloader, validloader)
            metric_list.append(metric)
            print('attach {}th in {}, metric {}'.format(attack_id, opt.aug_list, metric))

    pathname = 'search/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
    root_dir = os.path.dirname(pathname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if len(metric_list) > 0:
        print(np.mean(metric_list))
        np.save(pathname, metric_list)

    # maybe need old_state_dict
    model.load_state_dict(old_state_dict)
    score_list = list()
    for run in range(10):
        large_samle_list = [200 + run  * 100 + i for i in range(100)]
        score = accuracy_metric(large_samle_list, model, loss_fn, trainloader, validloader)
        score_list.append(score)
    
    print('time cost ', time.time() - start)
    
    pathname = 'accuracy/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
    root_dir = os.path.dirname(pathname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    np.save(pathname, score_list)
    print(score_list)



if __name__ == '__main__':
    main()
