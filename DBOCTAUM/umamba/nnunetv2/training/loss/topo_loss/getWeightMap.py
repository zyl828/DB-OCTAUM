import os
import sys

import gc
import math
import numpy as np
import cv2
import torch
import time
from skimage import measure
from skimage import io
import scipy.ndimage as ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from typing import Callable, Iterable, List, Set, Tuple

from nnunetv2.training.loss.topo_loss.utils.utils_overlap import OverlapTile


class SkeletonAwareWeight():
    """
    Skeleton Aware Weight

    """

    def __init__(self, eps=1e-20):
        """
        At this time, the weight is only suited to binary segmentation, so class_num = 2
        """
        self._class_num = 2
        self._eps = eps
        self.method = None
        self.overlap_tile = OverlapTile()  # speed up

    def _get_weight(self, mask: np.ndarray, method='skeaw_dilate_step20_iter2_bort', single_border=True) -> np.ndarray:
        """
        Get skeleton aware weight map
        :param mask: binary gt mask with shape (H, W)
        :return weight:  weight map with shape (H, W, 2)
        iter超参数，用于控制膨胀的程度
        对于原始 mask 中的每个像素位置 (i, j)，现在都有一个二维权重向量与之对应
        对于每个像素x，比较它在背景上的加权值ws0(x)与在膨胀后的前景上的加权值。如果背景加权值小于膨胀前景的加权值，那么该像素被判定为前景（苹果的一部分）；否则，被判定为背景假设有一个像素x，它紧挨着苹果的边缘。由于边界预测可能不完全准确，直接使用p0(x)和p1(x)可能会将该像素错误地分类为背景。但是，通过使用膨胀掩码md(x)，并且考虑到diter的适当值，即使预测的边界稍微偏离了真实边界，该像素也会被包含在膨胀后的前景区域内，从而被正确地分类为前景。
        """
        self.method = method
        # Get class weight for two channels
        weight = np.zeros((mask.shape[0], mask.shape[1], 2))
        class_weight = np.zeros((self._class_num, 1))
        for class_idx in range(self._class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight

        # Get weight for each channel
        for class_idx in range(self._class_num):
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == class_idx] = 1.0
            dis_trf = ndimage.distance_transform_edt(temp_mask)

            st = time.time()
            if class_idx == 1:
                # Get weight for border
                if single_border:
                    temp_weight = 1.0
                else:
                    label_map, label_num = measure.label(temp_mask, connectivity=1, background=1, return_num=True)
                    # temp_weight = self._get_border_weight(class_weight[class_idx, 0], temp_mask, dis_trf, label_map,
                    #                                       label_num)
                     # speed up
                    crop_images = self.overlap_tile.crop(temp_mask)
                    temp = []
                    for crop_mask in crop_images:
                        label_map, label_num = measure.label(crop_mask, connectivity=1, background=1, return_num=True)
                        temp_weight = self._get_border_weight(class_weight[class_idx, 0], crop_mask, dis_trf, label_map, label_num)
                        temp.append(temp_weight)
                    temp_weight = self.overlap_tile.stitch(temp)
            else:
                # Get weight for objects
                label_map, label_num = measure.label(temp_mask, connectivity=1, background=0, return_num=True)
                temp_weight = self._get_object_weight(class_weight[class_idx, 0], dis_trf, label_map, label_num)
            ed = time.time()
            # print("class:{}, time {:.4f}".format(class_idx, ed - st))
            weight[:, :, class_idx] = temp_weight * temp_mask
        return weight

    def _get_border_weight(self, wc: float, mask: np.ndarray, dis_trf: np.ndarray, label_map: np.ndarray,
                           label_num: int) -> np.ndarray:
        """
        Get border weight
        :param wc: class weight of border channel
        :param mask: real mask with shape (H,W), the border pixels equal 0 and the object pixels equal 1
        :param dis_trf: distance transform of border (shape (H,W)), it means the distance of each border pixel to the nearest object
        :param label_map: train_labels map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of border with shape (H, W)
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        h, w = mask.shape[:2]
        min_dis = np.ones((h, w, label_num))

        # For each connected region, calculate the closest distance from other pixels to this region.
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0
            # mask this connected region by set its value to 0
            mask_label = np.ones_like(mask)
            mask_label[min_row: max_row, min_col: max_col] -= bool_sub
            # Get the distance transform to masked connect component
            min_dis[:, :, label_idx] = ndimage.distance_transform_edt(mask_label) * mask

        # Calculate for each pixel the distances to the first nearest(d1) and second nearest(d2) connected regions
        min_dis.sort(axis=2)
        min_dis = min_dis[:, :, 0:2]
        # Calculate weight map by d1 and d2 for each pixel
        max_dis_trf = np.amax(dis_trf)
        weight = 1.0 + (
                    (2 * max_dis_trf - min_dis[:, :, 0] - min_dis[:, :, 1] + self._eps) / (2 * max_dis_trf + self._eps))

        weight[weight < 0] = 0.0
        return weight

    def _get_object_weight(self, wc: float, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get object weight
        :param wc: class weight of border channel
        :param dis_trf: distance transform of object (shape (H,W)), it means the distance of each pixel to the nearest border
        :param label_map: train_labels map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of object with shape (H, W)
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)

        # For each connect component, calculate its weight by its skeleton
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") / 255  # Lee Skeleton method
            if np.count_nonzero(bool_sub_sk == 1.0) == 0:
                # If there is no skelenton pixel, continue
                continue
            # Get the distance transform of skeleton pixel
            dis_trf_sk_sub = dis_trf[min_row: max_row, min_col: max_col] * bool_sub_sk

            # Get the distance transform to skeleton pixel
            indices = np.zeros(((np.ndim(bool_sub_sk),) + bool_sub_sk.shape), dtype=np.int32)
            dis_trf_to_sk = ndimage.distance_transform_edt(1 - bool_sub_sk, return_indices=True, indices=indices)

            h, w = bool_sub.shape[:2]
            dis_sk_map = np.ones((h, w, 2))
            dis_sk_map[:, :, 0] = dis_trf_to_sk  # d0
            dis_sk_map[:, :, 1] = dis_trf_sk_sub[indices[0, :, :], indices[1, :, :]]  # d1

            # Rectify, enusre d0 <= d1, d0: the distance of pixel to nearest skeleton pixel, d1: the distance d1 of nearest skeleton pixel to border
            dis_sk_map[:, :, 0][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]] = dis_sk_map[:, :, 1][
                dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]]

            weight_sub = 1 - (dis_sk_map[:, :, 0] / (dis_sk_map[:, :, 1] + self._eps))

            weight[min_row: max_row, min_col: max_col] += weight_sub * bool_sub

        return weight


if __name__ == "__main__":
    mask = np.random.randint(2, size=(320, 320, 1))
    weight_fuc = SkeletonAwareWeight()
    weight = weight_fuc._get_weight(mask)


    print('done!')