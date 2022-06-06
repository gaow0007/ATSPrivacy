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
parser.add_argument('--num_samples', default=5, type=int, help='Images per class')
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

def similarity_measures(img_batch, ref_batch, batched=True, method='fsim'):
        
    from image_similarity_measures.quality_metrics import fsim, issm, rmse, sam, sre, ssim, uiq
    methods = {'fsim':fsim, 'issm':issm, 'rmse':rmse, 'sam':sam, 'sre':sre, 'ssim':ssim, 'uiq':uiq }

    def get_similarity(img_in, img_ref):
        return methods[method](img_in.permute(1,2,0).numpy(), img_ref.permute(1,2,0).numpy())
        
    if not batched:
        sim = get_similarity(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        sim_list = []
        for sample in range(B):
            sim_list.append(get_similarity(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))

        sim_list = np.array(sim_list)
        sim_list = sim_list[~np.isnan(sim_list)]
        sim = np.mean(sim_list)
    return sim

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

    compute_privacy_score = True
    compute_acc_score = True

    sample_list = {}
    if opt.data == 'cifar100':
        num_classes = 100
    elif opt.data == 'FashionMinist':
        num_classes = 10
    for i in range(num_classes):
        sample_list[i] = []
    for idx, (_, label) in enumerate(validloader.dataset):   
        sample_list[label].append(idx)

    if compute_privacy_score:
        num_samples = opt.num_samples
        for label in range(num_classes):
            metric = []
            for idx in range(num_samples):
                metric.append(reconstruct(sample_list[label][idx], model, loss_fn, trainloader, validloader))
                print('attach {}th in class {}, auglist:{} metric {}'.format(idx, label, opt.aug_list, metric))
            metric_list.append(np.mean(metric,axis=0))

        pathname = 'search/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
        root_dir = os.path.dirname(pathname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if len(metric_list) > 0:
            print(np.mean(metric_list))
            np.save(pathname, metric_list)
    if compute_acc_score:
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
