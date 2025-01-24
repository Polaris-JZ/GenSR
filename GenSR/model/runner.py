import random
import numpy as np
import torch
import logging
import sys
import os
import math

def maybe_autocast(dtype=torch.float16):
    return torch.cuda.amp.autocast(dtype=dtype)

def optimizer(model, args):
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()
    if args.rank == 0:
        logging.info("number of trainable parameters: %d" % num_parameters)
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(1e-3),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    beta2 = 0.999
    _optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(args.base_lr),
        weight_decay=float(1e-3),
        betas=(0.9, beta2),
    )

    return _optimizer

def lr_scheduler(optimizer, args):
    """
    A property to get and create learning rate scheduler by split just in need.
    """

    max_epoch = args.base_num_epochs
    min_lr = args.base_min_lr
    init_lr = args.base_lr

    # optional parameters
    decay_rate = args.base_weight_decay
    warmup_start_lr = args.base_warmup_lr
    warmup_steps = args.base_warmup_steps
    iters_per_epoch = args.base_iter_per_epoch

    _lr_sched = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=max_epoch,
        iters_per_epoch=iters_per_epoch,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_rate=decay_rate,
        warmup_start_lr=warmup_start_lr,
        warmup_steps=warmup_steps,
    )

    return _lr_sched
    

class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr