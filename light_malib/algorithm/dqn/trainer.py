# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from light_malib.registry import registry
from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator
)
from .loss import DQNLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..return_compute import compute_return
from ..common.trainer import Trainer

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

@registry.registered(registry.TRAINER)
class DQNTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.id=tid
        self._loss = DQNLoss()

    def optimize(self, batch, **kwargs):
        total_opt_result = defaultdict(lambda: 0)
        policy = self.loss.policy
        
        global_timer.record("move_to_gpu_start")
        # move data to gpu
        for key,value in batch.items():
            if isinstance(value,np.ndarray):
                value=torch.FloatTensor(value)
            batch[key]=value.to(policy.device)
        global_timer.time("move_to_gpu_start","move_to_gpu_end","move_to_gpu")
        
        assert not policy.custom_config["use_rnn"]
        num_mini_batch=policy.custom_config["num_mini_batch"]
        data_generator_fn = functools.partial(
            simple_data_generator, batch, num_mini_batch, policy.device, shuffle=False
        )
            
        data_iter=data_generator_fn()
        for i in range(num_mini_batch):
            global_timer.record("data_generator_start")
            mini_batch=next(data_iter)
            global_timer.time("data_generator_start","data_generator_end","data_generator")
            global_timer.record("loss_start")
            tmp_opt_result = self.loss(mini_batch)
            global_timer.time("loss_start","loss_end","loss")
            for k, v in tmp_opt_result.items():
                total_opt_result[k] = v

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        pass