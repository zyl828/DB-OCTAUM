import math
import torch
import torch.nn.functional as F
from torch import nn
# from umamba.nnunetv2.nets.othernets.Sinkhorn import SinkhornDistance

def dotmat(X, Y, div=1):
  return  - X.bmm(Y.transpose(1, 2)) / div

''''ELA'''
class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7, device='cuda'):
        super(ELA, self).__init__()
        self.device = device
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False).to(device)
        self.gn = nn.GroupNorm(16, channel).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * x_h * x_w

''''CBAM'''
class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out
class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x
class CBAM(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattn = ChannelAttention(channel, scaling=scaling)
        self.spatialattn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattn(x)
        x = x * self.spatialattn(x)
        return x

'''Self-Attnetion'''
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (b, h*w, d)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (batch_size, channels//8, height*width)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (batch_size, channels, height*width)
        # Attention map
        energy = torch.bmm(query, key)  # (batch_size, height*width, height*width)
        attention = self.softmax(energy)  # Softmax to get attention weights
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (batch_size, channels, height*width)
        # Reshape back to original image shape
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


## GSA
class PAM_Module(nn.Module):
    """ GSA 空间注意力模块"""

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

## OTSA
class OTSA_Module(nn.Module):
    """ OTSA模块"""

    def __init__(self, in_dim, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.pos = PositionEmbeddingLearned(in_dim // factor)
        self.sab = SinkhornAttention(in_dim)
        # self.sab = SinkhornAttentionWithConv(in_dim)

    def forward(self, x):
        x_pos = self.pos(x)
        x = x + x_pos
        x = self.sab(x)

        return x


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的位置编码
    """
    def __init__(self, num_pos_feats=256, len_embedding=32):
        super().__init__()
        self.row_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.col_embed = nn.Embedding(len_embedding, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos


class ScaledDotProductAttention(nn.Module):
    """自注意力模块"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)  # b d n
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)          # b n d
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output

class SinkhornAttention(nn.Module):
    """Sinkhorn注意力"""

    def __init__(self, in_dims, attn_dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.sinkhorn = SinkhornDistance(eps=1, max_iter=1, cost=dotmat)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        sqrtV = math.sqrt(math.sqrt(d))

        q = x.view(m_batchsize, d, -1).permute(0, 2, 1) # b n c
        k = x.view(m_batchsize, d, -1).permute(0, 2, 1) # b n c
        v = x.view(m_batchsize, d, -1).permute(0, 2, 1) # b n c
        attn, _ , _ , _ = self.sinkhorn(q / sqrtV , k / sqrtV )  # b n n

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        output = torch.matmul(attn, v).permute(0, 2, 1) # b n n * b n c = b n c -> b c n
        output = output.view(m_batchsize, d, height, width)

        return output


class SinkhornAttentionWithConv(nn.Module):
    """Sinkhorn注意力"""

    def __init__(self, in_dims, attn_dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(attn_dropout)
        self.sinkhorn = SinkhornDistance(eps=1, max_iter=1, cost=dotmat)
        self.conv = nn.Conv2d(in_channels=in_dims, out_channels=in_dims * 3, kernel_size=1)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        sqrtV = math.sqrt(math.sqrt(d))

        qkv = self.conv(x)
        q, k, v = torch.split(qkv, d, dim=1)
        q = q.view(m_batchsize, d, -1).permute(0, 2, 1)  # b n c
        k = k.view(m_batchsize, d, -1).permute(0, 2, 1)  # b n c
        v = v.view(m_batchsize, d, -1).permute(0, 2, 1)  # b n c
        attn, _, _, _ = self.sinkhorn(q / sqrtV, k / sqrtV)  # b n n

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        output = torch.matmul(attn, v).permute(0, 2, 1)  # b n n * b n c = b n c -> b c n
        output = output.view(m_batchsize, d, height, width)

        return output

if __name__ == "__main__":
    x = torch.randn(4, 512, 32, 32)
    attn = OTSA_Module(512)
    # attn = PAM_Module(512)
    y = attn(x)
    print(y.shape)