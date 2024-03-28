
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
        train_dataset = self.dataset_cls(shuffle=True)
        return self.build_dataloader(train_dataset, shuffle=True)

    @data_loader
    def val_dataloader(self):
        train_dataset = self.dataset_cls(shuffle=False)
        return self.build_dataloader(train_dataset)

    @data_loader
    def test_dataloader(self):
        train_dataset = self.dataset_cls(shuffle=False)
        return self.build_dataloader(train_dataset, batch_by_size=False)

    def build_dataloader(self, dataset, shuffle=False, batch_by_size=True):
        indices = dataset.ordered_indices()

        batch_sampler = []
        if batch_by_size:
            mini_batch = []
            for i in range(0, len(indices)):
                mini_batch.append(indices[i])
                if len(mini_batch) == hparams['batch_size']:
                    batch_sampler.append(mini_batch.copy())
                    mini_batch.clear()
            if len(mini_batch) > 0:
                batch_sampler.append(mini_batch.copy())
        else:
            for i in range(0, len(indices)):
                batch_sampler.append([indices[i]])

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
        else:
            batches = batch_sampler


        num_workers = dataset.num_workers
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batches,
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
        # sample {'input_frames': (b, 9, h, w), 'gt_frames': (b, 20, h, w)}
        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        return total_loss, loss_output

    def run_model(self, model, sample, return_output=False):
        
        input_frames = sample['input_frames'].float()
        output = model(input_frames)
        losses = {}

        gt = sample['gt_frames'].float()[..., :1]
        losses['mse'] = self.mse_loss(output, gt)
        if not return_output:
            return losses
        else:
            return losses, output

    def mse_loss(self, output, target):
        assert output.shape == target.shape
        mse_loss = F.mse_loss(output, target)
        return mse_loss

    def on_train_end(self):
        self.trainer.save_checkpoint(epoch=self.current_epoch)

    ######################
    # testing
    ######################

    def test_start(self):
        self.gen_dir = os.path.join(hparams['work_dir'],
                                    f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)


    ######################
    # validation
    ######################

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: output: {"losses": {...}, "total_loss": float, ...} or (total loss: torch.Tensor, loss_log: dict)
        """
        # raise NotImplementedError
        outputs = {}
        losses, output = self.run_model(self.model, sample, return_output=True)
        output = output.detach().cpu().numpy()
        gt = torch.cat((sample['input_frames'], sample['gt_frames']), dim=1).detach().cpu().numpy()
        res_path = self.gen_dir

        def save_plots(field, labels, res_path, figsize=None,
                       vmin=0, vmax=10, cmap="viridis", npy=False, **imshow_args):

            for i, data in enumerate(field):
                #data(h,w,2),h=384,w=384 for normal;
                fig = plt.figure(figsize=figsize)
                ax = plt.axes()
                ax.set_axis_off()
                alpha = data[..., 0] / 1
                alpha[alpha < 1] = 0
                alpha[alpha > 1] = 1

                img = ax.imshow(data[..., 0], alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
                plt.savefig('{}/{}.png'.format(res_path, labels[i]))
                plt.close()  
                if npy:
                    with open( '{}/{}.npy'.format(res_path, labels[i]), 'wb') as f:
                        np.save(f, data[..., 0])


        data_vis_dict = {
            'radar': {'vmin': 1, 'vmax': 40},
        }
        vis_info = data_vis_dict['radar']

        if batch_idx <= hparams['num_save_samples']:
            path = os.path.join(res_path, str(batch_idx))
            os.makedirs(path, exist_ok=True)
            if hparams['case_type'] == 'normal':
                test_ims_plot = gt[0][:-2, 256-192:256+192, 256-192:256+192]
                img_gen_plot = output[0][:-2, 256-192:256+192, 256-192:256+192]
            else:
                test_ims_plot = gt[0][:-2]
                img_gen_plot = output[0][:-2]
            save_plots(test_ims_plot,
                       labels=['gt{}'.format(i + 1) for i in range(hparams['total_length'])],
                       res_path=path, vmin=vis_info['vmin'], vmax=vis_info['vmax'])
            save_plots(img_gen_plot,
                       labels=['pd{}'.format(i + 1) for i in range(9, hparams['total_length'])],
                       res_path=path, vmin=vis_info['vmin'], vmax=vis_info['vmax'])
        outputs['losses'] = losses
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        return outputs
