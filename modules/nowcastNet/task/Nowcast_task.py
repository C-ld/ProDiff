
from tasks.base_task import BaseTask
from tasks.base_task import data_loader, BaseConcatDataset
from modules.nowcastNet.models.nowcastnet import NowcastNet

from utils.hparams import hparams
from utils import audio
from utils.common_schedulers import RSQRTSchedule, NoneSchedule

from tqdm import tqdm
from modules.nowcastNet.task.dataset_utils import NowcastDataset

import json
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F
import utils
import os
import numpy as np
import filecmp
import pandas as pd

class NowcastTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = NowcastDataset
        
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}

    ######################
    # build model, dataloaders, optimizer, scheduler
    ######################

    def build_model(self):
        self.model = NowcastNet(hparams)
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.num_params(self.model, print_out=True, model_name="Nowcast")
        return self.model

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(shuffle=False)
        return self.build_dataloader(train_dataset)
        # train_dataset = self.dataset_cls(shuffle=True)
        # return self.build_dataloader(train_dataset, True, self.max_sentences, hparams['endless_ds'])


    @data_loader
    def val_dataloader(self):
        # raise NotImplementedError
        train_dataset = self.dataset_cls(shuffle=False)
        return self.build_dataloader(train_dataset)

    @data_loader
    def test_dataloader(self):
        raise NotImplementedError
        # test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        # self.test_dl = self.build_dataloader(
        #     test_dataset, False, self.max_valid_tokens,
        #     self.max_valid_sentences, batch_by_size=False)
        # return self.test_dl

    def build_dataloader(self, dataset, shuffle=False, required_batch_size_multiple=-1):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        indices = dataset.ordered_indices()

        batch_sampler = []
        for i in range(0, len(indices)):
            batch_sampler.append(indices[i])

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
        else:
            batches = batch_sampler


        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        batch_sampler = []
        for i in range(0, len(batches)):
            batch_sampler.append([batches[i]])
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=num_workers)
        # return torch.utils.data.DataLoader(dataset,
        #                                    batch_size=4,
        #                                    shuffle=True,
        #                                    num_workers=num_workers)

    def build_scheduler(self, optimizer):
        #schedule the learning rate
        # TODO lr change from 0.001 to 0.0001
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer)
        else:
            return NoneSchedule(optimizer)

    def build_optimizer(self, model):
        # TODO Adam + L2 regular VS AdamW, which better?
        self.optimizer = optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams['lr'])
        # self.optimizer = optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=hparams['lr'],
        #     betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
        #     weight_decay=hparams['weight_decay'])
        return optimizer

    ######################
    # training
    ######################
    def on_train_start(self):
        pass

    def on_epoch_start(self):
        self.training_losses_meter = {'total_loss': utils.AvgrageMeter()}

    def _training_step(self, sample, batch_idx, optimizer_idx):
        """

        :param sample:
        :param batch_idx:
        :return: total loss: torch.Tensor, loss_log: dict
        """
        # print(sample)
        # raise NotImplementedError
        # TODO:ld implement the train_step
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output



    def run_model(self, model, sample, return_output=False):
        
        input_frames = sample
        output = model(input_frames)
        losses = {}
        losses['mse'] = self.mse_loss(output, sample)
        # self.add_mel_loss(output['mel_out'], target, losses)
        # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def mse_loss(self, output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert output.shape == target.shape
        mse_loss = F.mse_loss(output, target, reduction='none')
        # weights = self.weights_nonzero_speech(target)
        # mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss
