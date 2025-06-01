import os
import sys
import numpy as np
import torch
import torch.nn as nn

from skimage import morphology, measure
from nnunetv2.training.loss.topo_loss.getWeightMap import SkeletonAwareWeight
from nnunetv2.training.loss.topo_loss.topoLoss import WeightMapBortLoss



class skeaTopoLoss(nn.Module):
    def __init__(self):
        super(skeaTopoLoss, self).__init__()
        self.wmp_criterion = WeightMapBortLoss()

    def dilate(self, a, k=3):
        if k == 0:
            return a
        return morphology.dilation(a, morphology.square(k))

    def calc_back_skelen(self, a):
        skelen = np.zeros_like(a)
        #将输入图像中的连通区域（即像素值相同且彼此相邻的区域）分配一个唯一的标签（整数）。这个标签用于标识图像中的不同对象或区域。
        mask_label, label_num = measure.label(a, connectivity=1, background=1, return_num=True)  # 先标记
        #获取mask_label图像中每个连通区域的属性集。
        image_props = measure.regionprops(mask_label, cache=False)  # 获得区域属性

        for li in range(label_num):
            image_prop = image_props[li]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") / 255  # Lee Skeleton method
            bool_sub_sk = bool_sub_sk.astype(np.int64)
            skelen[min_row:max_row, min_col:max_col] += bool_sub_sk

        back_skelen = self.dilate(skelen)
        fore_skelen = morphology.skeletonize(a, method="lee") / 255
        back_skelen = back_skelen[:, :, np.newaxis]
        fore_skelen = fore_skelen[:, :, np.newaxis]
        skelens = np.concatenate((back_skelen, fore_skelen), axis=2)

        return skelens


    def forward(self, pred, target, epoch_num):

        tar = target.squeeze(1).cpu().numpy()
        batch_size, height, width = tar.shape
        weights = []
        skelens = []
        class_weights = []

        for i in range(batch_size):
            # 提取批次中的单张图像
            single_target = tar[i:i + 1, :, :]

            weight_fuc = SkeletonAwareWeight()
            weight = weight_fuc._get_weight(single_target[0, :, :])
            weights.append(torch.from_numpy(weight))

            skelen = self.calc_back_skelen(single_target[0, :, :])
            skelens.append(torch.from_numpy(skelen))

            class_num = 2
            class_weight = np.zeros((class_num, 1))
            for class_idx in range(class_num):
                idx_num = np.count_nonzero(single_target == class_idx)
                class_weight[class_idx, 0] = idx_num
            min_num = np.amin(class_weight)
            class_weight = class_weight * 1.0 / min_num
            class_weight = np.sum(class_weight) - class_weight
            class_weights.append(torch.from_numpy(class_weight))


        all_weights = torch.stack(weights, dim=0)
        all_skelens = torch.stack(skelens, dim=0)
        all_class_weights = torch.stack(class_weights, dim=0)
 
        skew_loss = self.wmp_criterion(pred, target, all_weights, all_class_weights, label_skelen=all_skelens,  epoch=epoch_num)

        return skew_loss