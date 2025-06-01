import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
from torch import nn
import torchvision
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.nets.UMambaEncSS2D_2d import get_umamba_enc_2d_from_plans as get_umamba_encss2d_2d_from_plans

class getRef(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        # model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
        #                                      num_input_channels, deep_supervision=enable_deep_supervision)
        # model = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
        #                                      num_input_channels, deep_supervision=enable_deep_supervision)
        # model = get_umamba_encss2d_2d_from_plans(plans_manager, dataset_json, configuration_manager,
        #                                      num_input_channels, deep_supervision=enable_deep_supervision)
        model = get_umamba_esnew_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)

        print("Model: {}".format(model))
        return model


    def eval_step(self, batch: dict) -> dict:
        img_dir = "/root/autodl-tmp/U-Mamba-main/data/nnUNet_raw/Dataset120_OCTASegmentation/imagesTs/"
        images = os.listdir(img_dir)
        #chk = torch.load('/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainer__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        chk = torch.load('/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainerUMambaES_New__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        # chk = torch.load(
        #     '/root/autodl-tmp/U-Mamba-main/data/nnUNet_results/Dataset110_OCTASegmentation/nnUNetTrainerUMambaEncNoAMP__nnUNetPlans__2d/fold_9/checkpoint_best.pth')
        self.network.state_dict(chk)
        model = self.network.eval()
        # input_H, input_W = 512, 512
        input_H, input_W = 320, 320
        heatmap = np.zeros([input_H, input_W])
        layer = model.encoder.stages[0]
        # layer = model.encoder.stages[1]
        # layer = model.encoder.stages[6]
        # layer = model.decoder.stages[1]
        # layer = model.decoder.stages[5]
        # layer = model.decoder.upsample_layers[5]
        # layer = model.decoder.transpconvs[5]
        # layer = model.decoder.seg_layers[5]
        # layer = model.encoder.mamba_layers[2].mamba.out_proj
        print(layer)

        def farward_hook(module, data_input, data_output):
            fmap_block.append(data_output)
            input_block.append(data_input)

        for img in images:
            read_img = os.path.join(img_dir, img)
            image = Image.open(read_img)

            image = image.resize((input_H, input_W))
            image = np.float32(image) / 255
            input_tensor = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(image)

            # 添加batch维度
            input_tensor = input_tensor.unsqueeze(0)

            if torch.cuda.is_available():
                model = model.cuda()
                input_tensor = input_tensor.cuda()

            input_tensor.requires_grad = True
            fmap_block = list()
            input_block = list()

            layer.register_forward_hook(farward_hook)
            output = model(input_tensor)
            feature_map = fmap_block[0].mean(dim=1, keepdim=False).squeeze()
            feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)].backward(retain_graph=True)
            grad = torch.abs(input_tensor.grad)
            grad = grad.mean(dim=1, keepdim=False).squeeze()
            heatmap = heatmap + grad.cpu().numpy()

        cam = heatmap

        # 对累加的梯度进行归一化
        cam = cam / cam.max()

        # 可视化，蓝色值小，红色值大
        cam = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(cam)
        ax.axis('off')  # 关闭坐标轴
        plt.show()
        plt.savefig("test/es_test_e0.png")  # 保存每个图像的热图，或者进行其他处理
        print('cam end')
