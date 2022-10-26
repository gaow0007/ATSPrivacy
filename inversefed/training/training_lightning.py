from asyncio.log import logger
from math import gamma
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from collections import defaultdict
from .scheduler import GradualWarmupScheduler

from torch.distributions.dirichlet import Dirichlet
from torch.nn.functional import one_hot
from torch.utils.data.dataset import Dataset
import numpy as np


def prune(gradient, percent):
    k = int(gradient.numel() * percent * 0.01)
    shape = gradient.shape
    gradient = gradient.flatten()
    index = torch.topk(torch.abs(gradient), k, largest=False)[1]
    gradient[index] = 0.
    gradient = gradient.view(shape)
    return gradient


def add_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * torch.randn(param.grad.data.shape).cuda()


def lap_sample(shape):
    from torch.distributions.laplace import Laplace
    m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    return m.expand(shape).sample()

def lap_noise(model, lr):
    for param in model.parameters():
        param.grad.data += lr * lap_sample(param.grad.data.shape).cuda()


def global_prune(model, percent):
    for param in model.parameters():
        param.grad.data = prune(param.grad.data, percent)


class LitModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, scheduler, opt):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt    = opt

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward

        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            imgs, labels = batch
            preds = self.model(imgs)
            loss = self.loss_fn(preds, labels)
            predicts = torch.sigmoid(preds)
            predicts = torch.round(predicts) 
            acc = (predicts == labels).float().mean()
             
        else: 
            if isinstance(batch, dict):
                model_output = self.model(**batch)
                loss = model_output.loss
                labels = batch['labels']
                preds = model_output.logits 
            else:
                imgs, labels = batch
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('train_acc', acc)
        self.log('train_loss', loss)
        return loss
    
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/8866
    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
    
    
        if self.opt.defense is not None:
            if 'gaussian' in self.opt.defense:
                if '1e-3' in self.opt.defense:
                    add_noise(self.model, 1e-3)
                elif '1e-2' in self.opt.defense:
                    add_noise(self.model, 1e-2)
                else:
                    raise NotImplementedError
            elif 'lap' in self.opt.defense:
                if '1e-3'  in self.opt.defense:
                    lap_noise(self.model, 1e-3)
                elif '1e-2' in self.opt.defense:
                    lap_noise(self.model, 1e-2)
                elif '1e-1' in self.opt.defense:
                    lap_noise(self.model, 1e-1)
                else:
                    raise NotImplementedError
            
            elif 'prune' in self.opt.defense:
                found = False
                for i in [10, 20, 30, 50, 70, 80, 90, 95, 99]:
                    if str(i) in self.opt.defense:
                        found=True
                        global_prune(self.model, i)

                if not found:
                    raise NotImplementedError


    def validation_step(self, batch, batch_idx):

        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            imgs, labels = batch
            preds = self.model(imgs)
            loss = self.loss_fn(preds, labels)
            predicts = torch.sigmoid(preds)
            predicts = torch.round(predicts) 
            acc = (predicts == labels).float().mean()
        else:
            if isinstance(batch, dict):
                model_output = self.model(**batch)
                loss = model_output.loss
                labels = batch['labels']
                preds = model_output.logits 
            else:
                imgs, labels = batch
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('vaild_acc', acc)
        self.log('vaild_loss', loss)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]



def set_optimizer(model, defs):
    """Build model optimizer and scheduler from defs.

    The linear scheduler drops the learning rate in intervals.
    # Example: epochs=160 leads to drops at 60, 100, 140.
    """
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
        # Scheduler is fixed to 120 epochs so that calls with fewer epochs are equal in lr drops.
    if defs.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    if defs.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=10, after_scheduler=scheduler)

    return optimizer, scheduler

def train_pl(model, loss_fn, trainloader, validloader, defs, setup=dict(dtype=torch.float, device=torch.device('cpu')), save_dir=None, opt=None):
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir)
    trainer = pl.Trainer(gpus=1,
                    max_epochs=defs.epochs,
                    logger=tb_logger,
                    val_check_interval=0.2,
                    log_every_n_steps=50,
                    # log_every_n_steps=10, # for tiny imagenet training
                    callbacks=[
                        ModelCheckpoint(
                            save_top_k = 1,
                            dirpath=save_dir,
                            filename='{epoch:03d}-{vaild_acc:.4f}',
                            every_n_epochs = int(0.1 * defs.epochs),
                            save_last=True,
                            # save_on_train_epoch_end=True
                        )
                    ])
    optimizer, scheduler = set_optimizer(model, defs)
    model = LitModule(model, loss_fn, optimizer, scheduler, opt)

    is_instahide = False
    if is_instahide:
        instahid =  InstahideDefense(trainloader.dataset,
                        4,
                        0.65,
                        0,
                        torch.device('cuda:0'),
                        False)

        instahid.apply(model)
        
        batch_size = 128
        for batch_idx, (batch, labels) in enumerate(trainloader):
            trainloader.dataset[batch_idx*batch_size:(batch_idx+1)*batch_size] = instahid.do_instahide(batch, batch_idx)
        # batch, labels = instahid.do_instahide(batch, batch_idx)

    if len(os.listdir(save_dir)) > 0:
        import glob
        trainer.fit(model, 
                trainloader,
                validloader,
                ckpt_path=glob.glob(save_dir+'/*.ckpt')[-1],
        )
    else:
        trainer.fit(model, 
                trainloader,
                validloader,
        )

def validation(model, loss_fn, dataloader, defs, setup, stats, save_dir):
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    # save_dir = '/home/zx/New_atsp/ATSPrivacy-new/validation'
    from pytorch_lightning import loggers as pl_loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir)
    trainer = pl.Trainer(gpus=1,
                    max_epochs=defs.epochs,
                    logger=tb_logger,
                    )
  
    model = LitModule(model, loss_fn, optimizer=None, scheduler=None, opt=None)
    trainer.validate(model, dataloader)

class InstahideDefense():
    def __init__(self,
                 mix_dataset: Dataset,
                 klam: int,
                 upper_bound: float,
                 lower_bound: float,
                 device: torch.device = None,
                 use_csprng: bool = True,
                 cs_prng: torch.Generator = None):
        """
        Args:
            mix_dataset (Dataset): the original training dataset
            klam (int): the numebr of data points to mix for each encoding
            upper_bound (float): the upper bound for mixing coefficients
            lower_bound (float): the lower bound for mixing coefficients
            device (torch.device, optional): the device to run training on. Defaults to None.
            use_csprng (bool, optional): whether to use cryptographically secure pseudorandom number generator. Defaults to True.
            cs_prng (torch.Generator, optional): the cryptographically secure pseudorandom number generator. Defaults to None.
        """

        self.klam = klam
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.device = device
        self.alpha = [1.0] * klam
        self.alpha[0] = 3.0

        self.mix_dataset = mix_dataset
        self.x_values, self.y_values = None, None
        if isinstance(self.mix_dataset, torch.utils.data.Subset):
            all_classes = list(self.mix_dataset.dataset.classes)
        elif isinstance(self.mix_dataset, torch.utils.data.dataset.Dataset):
            all_classes = list(self.mix_dataset.classes)
        self.num_classes = len(all_classes)
        self.dataset_size = len(self.mix_dataset)

        self.lambda_sampler_single = Dirichlet(torch.tensor(self.alpha))
        self.lambda_sampler_whole = Dirichlet(
            torch.tensor(self.alpha).repeat(self.dataset_size, 1))
        self.use_csprng = use_csprng

        if self.use_csprng:
            if cs_prng is None:
                # self.cs_prng = csprng.create_random_device_generator()
                raise NotImplementedError
            else:
                self.cs_prng = cs_prng

    # @profile
    def generate_mapping(self, return_tensor=True):
        """Generate the mapping and coefficients for InstaHide

        Args:
            return_tensor (bool, optional): whether to return the results in the format of PyTorch tensor. Defaults to True.

        Returns:
            (numpy.array): the mapping and coefficients array
        """
        if not self.use_csprng:
            lams = np.random.dirichlet(alpha=self.alpha,
                                       size=self.dataset_size)

            selects = np.asarray([
                np.random.permutation(self.dataset_size)
                for _ in range(self.klam)
            ])
            selects = np.transpose(selects)

            for i in range(self.dataset_size):
                # enforce that k images are non-repetitive
                while len(set(selects[i])) != self.klam:
                    selects[i] = np.random.randint(0, self.dataset_size,
                                                   self.klam)
                if self.klam > 1:
                    while (lams[i].max() > self.upper_bound) or (
                            lams[i].min() <
                            self.lower_bound):  # upper bounds a single lambda
                        lams[i] = np.random.dirichlet(alpha=self.alpha)
            if return_tensor:
                return (
                    torch.from_numpy(lams).float().to(self.device),
                    torch.from_numpy(selects).long().to(self.device),
                )
            else:
                return np.asarray(lams), np.asarray(selects)

        else:
            lams = self.lambda_sampler_whole.sample().to(self.device)
            selects = torch.stack([
                torch.randperm(self.dataset_size,
                               device=self.device,
                               generator=self.cs_prng)
                for _ in range(self.klam)
            ])
            selects = torch.transpose(selects, 0, 1)

            for i in range(self.dataset_size):
                # enforce that k images are non-repetitive
                while len(set(selects[i])) != self.klam:
                    selects[i] = torch.randint(0,
                                               self.dataset_size,
                                               self.klam,
                                               generator=self.cs_prng)
                if self.klam > 1:
                    while (lams[i].max() > self.upper_bound) or (
                            lams[i].min() <
                            self.lower_bound):  # upper bounds a single lambda
                        lams[i] = self.lambda_sampler_single.sample().to(
                            self.device)
            if return_tensor:
                return lams, selects
            else:
                return np.asarray(lams), np.asarray(selects)

    def instahide_batch(
        self,
        inputs: torch.tensor,
        lams_b: float,
        selects_b: np.array,
    ):
        """Generate an InstaHide batch.

        Args:
            inputs (torch.tensor): the original batch (only its size is used)
            lams_b (float): the coefficients for InstaHide
            selects_b (np.array): the mappings for InstaHide

        Returns:
            (torch.tensor): the InstaHide images and labels
        """
        mixed_x = torch.zeros_like(inputs)
        mixed_y = torch.zeros((len(inputs), self.num_classes),
                              device=self.device)

        for i in range(self.klam):
            x = torch.index_select(self.x_values, 0, selects_b[:, i]).clone()
            ys_onehot = torch.index_select(self.y_values, 0,
                                           selects_b[:, i]).clone()
            # need to broadcast here to make row-wise multiplication work
            # see: https://stackoverflow.com/questions/53987906/how-to-multiply-a-tensor-row-wise-by-a-vector-in-pytorch
            mixed_x += lams_b[:, i][:, None, None, None] * x
            mixed_y += lams_b[:, i][:, None] * ys_onehot

        # Apply InstaHide random sign flip
        sign = torch.randint(2, size=list(mixed_x.shape),
                             device=self.device) * 2.0 - 1
        mixed_x *= sign.float()

        return mixed_x, mixed_y

    def apply(self, model: pl.LightningModule):

        self.cur_selects, self.cur_selects = None, None

        def regenerate_mappings(module):
            """Regenerate InstaHide mapping and coefficients at the begining of each epoch

            Args:
                module (pl.LightningModule): the pl.LightningModule for training
            """
            self.cur_lams, self.cur_selects = self.generate_mapping(
                return_tensor=True)

            # IMPORTANT! Use new augmentations for every epoch
            self.x_values = torch.stack([data[0] for data in self.mix_dataset
                                         ]).to(self.device)

            if self.y_values is None:
                self.y_values = torch.from_numpy(
                    np.asarray([data[1]
                                for data in self.mix_dataset])).to(self.device)
                if len(self.y_values.shape) == 1:
                    self.y_values = one_hot(
                        self.y_values, num_classes=self.num_classes).float()

        model._on_train_epoch_start_callbacks.append(
            regenerate_mappings)

        regenerate_mappings(model)

        # @profile
    def do_instahide(self, batch: torch.tensor, batch_idx: int, use_tensor=True, *args, **kwargs):
        """Run InstaHide for a given batch

        Args:
            batch ((torch.tensor)): the original batch (only the size information is used)
            batch_idx (int): index of the batch; used to slice the whole mapping array
            use_tensor (bool, optional): whether the mapping and coefficients are in the format of PyTorch tensor. Defaults to True.
        """
        inputs, targets = batch
        batch_size = len(inputs)
        start_idx = batch_idx * batch_size
        batch_device = batch[0].device
        if use_tensor:
            batch_indices = torch.arange(start_idx,
                                            start_idx + batch_size,
                                            device=self.device).long()
            lams_b = torch.index_select(self.cur_lams, 0, batch_indices)
            selects_b = torch.index_select(self.cur_selects, 0,
                                            batch_indices)
        else:
            batch_indices = range(start_idx, start_idx + batch_size)
            lams_b = (torch.from_numpy(
                np.asarray([self.cur_lams[i] for i in batch_indices
                            ])).float().to(batch_device))
            selects_b = (torch.from_numpy(
                np.asarray([self.cur_selects[i] for i in batch_indices
                            ])).long().to(batch_device))

        mixed_inputs, mixed_targets = self.instahide_batch(
            inputs, lams_b, selects_b)
        return (mixed_inputs, mixed_targets)
