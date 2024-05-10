from utils.hparams import hparams
from tasks.base_task import BaseDataset

import numpy as np
import os, cv2
import torch.optim
import torch.utils.data
import torch.distributions

class NowcastDataset(BaseDataset):
    def __init__(self, shuffle=False, mode="test"):
        super().__init__(shuffle)
        self.hparams = hparams
        # Maybe no need 
        self.input_data_type = hparams.get('input_data_type', 'float32')
        self.output_data_type = hparams.get('output_data_type', 'float32')
        self.img_width = hparams['img_width']
        self.img_height = hparams['img_height']
        self.input_length = hparams['input_length']
        self.length = hparams['total_length'] #29
        self.data_path = hparams['data_path'] #target dataset path like: /data/dataset/mrms/figure
        self.type = hparams['type'] #'test'
        self.is_using_old = hparams['if_using_old'] #italy: false

        self.case_list = [] # (num_of_cases, 29, ...)
        year_list = os.listdir(self.data_path)
        year_list.sort()
        #遍历数据集下目录，根据mode确定遍历范围，获取所有单个序列并加入case_list
        if mode == "test":
            # find the last year and get all runs
            print("mode: ", mode)
            last_year = year_list[-1]
            if len(last_year) == 4:
                ypath = self.data_path + '/' + last_year
                day_list = os.listdir(ypath)
                day_list.sort()
                for day in day_list:
                    dpath = ypath + '/' + day
                    corp_list = os.listdir(dpath)
                    corp_list.sort()
                    for corp in corp_list:
                        case = []
                        png_list = os.listdir(dpath + '/' + corp)
                        png_list.sort()
                        for png in png_list:
                            case.append(dpath + '/' + corp + '/' + png)
                        self.case_list.append(case)
            else:
                print("Error: test dataset not supported")
                raise Exception("test dataset not supported")
            
        elif mode == "train":
            # get all years data except the last year
            print("mode: ", mode)
            years = year_list[:-1]
            for year in years:
                if len(year) == 4:
                    ypath = self.data_path + '/' + year
                    day_list = os.listdir(ypath)
                    day_list.sort()
                    for day in day_list:
                        dpath = ypath + '/' + day
                        corp_list = os.listdir(dpath)
                        corp_list.sort()
                        for corp in corp_list:
                            case = []
                            png_list = os.listdir(dpath + '/' + corp)
                            png_list.sort()
                            for png in png_list:
                                case.append(dpath + '/' + corp + '/' + png)
                            self.case_list.append(case)
                else:
                    print("Error: train dataset not supported")
                    raise Exception("train dataset not supported")
                
        elif mode == "valid":
            # get all years 1st data except the last year
            print("mode: ", mode)
            years = year_list[:-1]
            for year in years:
                if len(year) == 4:
                    ypath = self.data_path + '/' + year
                    day_list = os.listdir(ypath)
                    day_list.sort()
                    for day in day_list[::30]:
                        dpath = ypath + '/' + day
                        corp_list = os.listdir(dpath)
                        corp_list.sort()
                        for corp in corp_list:
                            case = []
                            png_list = os.listdir(dpath + '/' + corp)
                            png_list.sort()
                            for png in png_list:
                                case.append(dpath + '/' + corp + '/' + png)
                            self.case_list.append(case)
                else:
                    print("Error: train dataset not supported")
                    raise Exception("train dataset not supported")
        print("case total: ", len(self.case_list))
        # for name in name_list:
        #     case = []
        #     for i in range(29):
        #         case.append(self.data_path + '/' + name + '/' + name + '-' + str(i).zfill(2) + '.png')
        #     self.case_list.append(case)

 
    def load(self, index):
        #获取第 index 个 case 的图片流
        data = []
        # print(self.case_list[index][0])
        for img_path in self.case_list[index]:
            img = cv2.imread(img_path, 2) 
            #以单通道灰度图的方式读取图像
            data.append(np.expand_dims(img, axis=0)) 
            #img(w,h)先扩展了一个维度变为(1,h,w),再加入data(n+1,1,h,w)
        data = np.concatenate(data, axis=0).astype(self.input_data_type)
        if self.is_using_old:
            data = data / 10.0 - 3.0 
        #↑把29个图解包合并，把每个单位作为32位浮点数做运算,data最终变为(29,h,w)
        assert data.shape[1]<=1024 and data.shape[2]<=1024 
        #检查 h<=1024,w<=1024
        return data

    def __getitem__(self, index):
        data = self.load(index)[-self.length:].copy()
        #data(29,h,w)

        mask = np.ones_like(data)
        # # 复制全1矩阵mask
        mask[data < 0] = 0
        data[data < 0] = 0
        # 数据和mask负数位都置0
        data = np.clip(data, 0, 128)
        # gt的所有>128的位置都置为128, <0的置为0
        vid = np.zeros((self.length, self.img_height, self.img_width, 2))
        vid[..., 0] = data
        vid[..., 1] = mask
        img = dict()
        
        img['input_frames'] = vid[:self.input_length]
        img['gt_frames'] = vid[self.input_length:]
        return img

    def __len__(self):
        return len(self.case_list) # num_of_cases

    # def collater(self, samples):
    #     if len(samples) == 0:
    #         return {}
    #     hparams = self.hparams
    #     id = torch.LongTensor([s['id'] for s in samples])
    #     item_names = [s['item_name'] for s in samples]
    #     text = [s['text'] for s in samples]
    #     txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
    #     mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
    #     txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
    #     mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

    #     batch = {
    #         'id': id,
    #         'item_name': item_names,
    #         'nsamples': len(samples),
    #         'text': text,
    #         'txt_tokens': txt_tokens,
    #         'txt_lengths': txt_lengths,
    #         'mels': mels,
    #         'mel_lengths': mel_lengths,
    #     }

    #     if hparams['use_spk_embed']:
    #         spk_embed = torch.stack([s['spk_embed'] for s in samples])
    #         batch['spk_embed'] = spk_embed
    #     if hparams['use_spk_id']:
    #         spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
    #         batch['spk_ids'] = spk_ids
    #     return batch
   