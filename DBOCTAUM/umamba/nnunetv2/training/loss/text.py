import ot
import numpy as np

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

if __name__ == '__main__':
    x = torch.randn(7,2,4,4)
    y = torch.randn(7,1,4,4)
    p = optimal_transport1(x,y)
    print(p.shape)