import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn, autocast
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
import ot
import numpy as np

def optimal_transport(posteriors, epsilon=0.1):
    """
    posteriors: U-Net 输出的像素概率分布，形状为 (batch_size, num_classes, height, width)
    epsilon: Sinkhorn 算法的正则化参数，控制平滑程度
    """
    batch_size, num_classes, height, width = posteriors.shape
    # 假设每个像素的概率均匀分布
    P = np.ones((height * width, num_classes)) / num_classes
    Q = posteriors.view(batch_size, num_classes, -1).permute(0, 2, 1).cpu().detach().numpy()  # 转置后的概率

    # 对每个批次的图像进行最优传输优化
    optimal_transport_maps = []
    for i in range(batch_size):
        C = np.linalg.norm(P - Q[i], axis=1)  # 计算成本矩阵，基于距离
        G = ot.sinkhorn(P, Q[i], C, epsilon)  # 计算最优传输矩阵
        optimal_transport_maps.append(G)

    # 最优传输后的结果（形状为 [batch_size, height, width, num_classes]）
    optimal_transport_maps = torch.tensor(np.array(optimal_transport_maps), device=posteriors.device)
    return optimal_transport_maps


def optimal_transport1(sources, target_labels, epsilon=0.1):
    """
    posteriors: U-Net 输出的像素概率分布，形状为 (batch_size, num_classes, height, width)
    target_labels: 目标标签，形状为 (batch_size, 1, height, width)，标签是类别索引
    epsilon: Sinkhorn 算法的正则化参数，控制平滑程度
    """
    batch_size, num_classes, height, width = sources.shape

    # Step 1: 将目标标签转换为 one-hot 编码
    target_labels = target_labels.long()  # 确保目标标签是长整型
    target_one_hot = torch.zeros(batch_size, num_classes, height, width, device=sources.device)

    # 使用 scatter_ 在 dim=1 维度上填充类别索引
    target_one_hot = target_one_hot.scatter_(1, target_labels, 1)  # 将目标标签转换为 one-hot 编码

    # Step 2: 将 target_one_hot 和 posteriors 转换为适合最优传输计算的格式
    # target_one_hot 转换为 (batch_size, num_classes, height*width)
    P = target_one_hot.view(batch_size, num_classes, -1).permute(0, 2, 1).cpu().detach().numpy()  # 目标标签的一维化
    Q = posteriors.view(batch_size, num_classes, -1).permute(0, 2, 1).cpu().detach().numpy()  # U-Net 输出的一维化

    # Step 3: 计算最优传输矩阵
    optimal_transport_maps = []
    for i in range(batch_size):
        # Step 3.1: 计算成本矩阵，基于 L2 范数
        C = np.linalg.norm(P[i] - Q[i], axis=1)  # 计算每个像素与类别的 L2 距离

        # Step 3.2: 使用 Sinkhorn 算法计算最优传输
        G = ot.sinkhorn(P[i], Q[i], C, epsilon)  # Sinkhorn 算法求解最优传输矩阵
        optimal_transport_maps.append(G)

    # Step 4: 将结果转换为 PyTorch 张量并返回
    optimal_transport_maps = torch.tensor(np.array(optimal_transport_maps), device=posteriors.device)
    return optimal_transport_maps

class nnUNetTrainerUMambaESink(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                             num_input_channels, deep_supervision=enable_deep_supervision)
        
        print("UMambaESink: {}".format(model))

        return model


    def train_step(self, batch: dict) -> dict:
        data = batch['data']  #[7,3,320,320]
        target = batch['target']#list 6,[7,1,10,10]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)

        #sinkhorn
        optimized_output = optimal_transport1(output[0], target[0])

        # l = self.loss(output, target)
        l = self.loss(optimized_output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        output = output[0]
        target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

