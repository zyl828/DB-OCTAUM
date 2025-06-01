import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.nets.UMambaEncSS2D_2d import get_umamba_enc_2d_from_plans as get_umamba_encss2d_2d_from_plans
import numpy as np
from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
from timm.utils import AverageMeter

class getHeatMap:
    def __init__(self, single_data, layer, model):
        self.data = single_data
        self.layer = layer
        self.net = model
        self.fmap_blocks = []
        self.input_blocks = []
        self.heatmaps = []
        self.handle = self.layer.register_forward_hook(self.forward_hook)

    def forward_hook(self,module,fea_in,fea_out):
        self.fmap_blocks.append(fea_out)

    def getMap(self):
        self.fmap_blocks.clear()  # 清空 fmap_blocks 以防止内存泄漏
        self.net.zero_grad()  # 重置梯度
        self.data.requires_grad_(True)  # 确保数据需要梯度
        output = self.net(self.data)
        feature_map = self.fmap_blocks[0].mean(dim=1,keepdim=False).squeeze()
        feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)
        grad = torch.abs(self.data.grad)
        grad = grad.mean(dim=1, keepdim=False).squeeze()
        return grad


class evalRef_new(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        # model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
        #                                      num_input_channels, deep_supervision=enable_deep_supervision)
        model = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                             num_input_channels, deep_supervision=enable_deep_supervision)
        # model = get_umamba_encss2d_2d_from_plans(plans_manager, dataset_json, configuration_manager,
        #                                      num_input_channels, deep_supervision=enable_deep_supervision)

        print("Model: {}".format(model))
        return model



    def eval_step(self, batch: dict) -> dict:

        data = batch['data']
        chk = torch.load(
            '/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainer__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        # chk = torch.load('/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainerUMambaEAP_SS2D__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        # chk = torch.load(
        #     '/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainerUMambaEncNoAMP__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        self.network.state_dict(chk)
        model = self.network.eval()

        # layer = model.decoder.upsample_layers[0]  #ss2d input
        layer = model.encoder.stages[6]  #encoder-1
        # layer = model.decoder.stages[1]
        # layer = model.decoder.seg_layers[1]

        data = data.to(self.device, non_blocking=True)
        data.requires_grad = True
        heatmapper = getHeatMap(data, layer, model)
        grad = heatmapper.getMap()

        return grad
