import os
import sys 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from skimage import measure
from skimage import morphology
import torch.nn as nn
from nnunetv2.training.loss.topo_loss.utils.utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


# show middle results
def show(pred_ori, mask_ori, tfn, ffn, tfp, ffp):
    pred = pred_ori
    mask = mask_ori
    #colors = ['white', 'silver', 'silver', 'k', 'k', 'k']
    colors = ['k', 'white','red']
    cmap1 = plt.get_cmap('plasma')  # inferno

    scmap = mcolors.ListedColormap(colors)

    colors2 = ['k', 'white', 'blue']
    scmap2 = mcolors.ListedColormap(colors2)

    colors3 = ['k', 'white', 'blue', 'red']
    scmap3 = mcolors.ListedColormap(colors3)

    plt.figure(figsize=(20, 10))

    flip_tf = pred + tfn - tfp

    plt.subplot(2,4,1), plt.imshow(mask, cmap='gray'), plt.axis("off"), plt.title('train_labels', fontsize=24)
    plt.subplot(2,4,2), plt.imshow(pred, cmap='gray'), plt.axis("off"), plt.title('pred', fontsize=24)  
    plt.subplot(2,4,5), plt.imshow(flip_tf, cmap='gray'), plt.axis("off"), plt.title('flip_tf', fontsize=24)  
    plt.subplot(2,4,6), plt.imshow(pred + 2*tfn + 2*tfp, cmap=scmap3), plt.axis("off"), plt.title('tfn+tfp', fontsize=24)  
    
    plt.subplot(2,4,3), plt.imshow(pred+2*tfn, cmap=scmap), plt.axis("off"), plt.title('TFN', fontsize=24)
    plt.subplot(2,4,4), plt.imshow(pred+tfp, cmap=scmap), plt.axis("off"), plt.title('TFP', fontsize=24)
    plt.subplot(2,4,7), plt.imshow(pred+2*ffn, cmap=scmap2), plt.axis("off"), plt.title('FFN', fontsize=24)  
    plt.subplot(2,4,8), plt.imshow(pred+ffp, cmap=scmap2), plt.axis("off"), plt.title('FFP', fontsize=24) 
          
    plt.show()


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def minus_torch(a, b):
    c = F.relu(a-b)
    return c


def calc_back_skelen_torch(x):
    B, H, W = x.size()
    x_b_skelen = torch.zeros((B, H, W)).to(x.device)
    x_f_skelen = torch.zeros((B, H, W)).to(x.device)
    x_np = x.cpu().detach().numpy()
    for i in range(B):
        a = x_np[i, :, :]
        skelen = np.zeros_like(a)
        mask_label, label_num = measure.label(a, connectivity=1, background=1, return_num=True) # train_labels it
        image_props = measure.regionprops(mask_label, cache=False) # get regions props

        fore_sk = morphology.skeletonize(a, method="lee") /255 # Lee Skeleton method   # fore skelen
        x_f_skelen[i, :, :] = torch.from_numpy(fore_sk).to(x.device)

        for li in range(label_num):
            image_prop = image_props[li]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = morphology.skeletonize(bool_sub, method="lee") /255 # Lee Skeleton method
            skelen[min_row:max_row, min_col:max_col] += bool_sub_sk.astype(np.int64)
            # for mid check
            #plt.figure(figsize=(20, 10))
            #plt.subplot(1,2,1), plt.imshow(bool_sub, cmap='gray'), plt.axis("off"), plt.title('bool_sub', fontsize=24)
            #plt.subplot(1,2,2), plt.imshow(bool_sub_sk, cmap='gray'), plt.axis("off"), plt.title('bool_sub_sk', fontsize=24)
            #plt.show()
        x_b_skelen[i, :, :] = torch.from_numpy(skelen).to(x.device)  # back skelen
    return soft_dilate(x_b_skelen.unsqueeze(1)).squeeze(1), x_f_skelen


# get connected regions
def get_tfn_conn_torch(fn, tfn):
    B, H, W = tfn.size()
    tfn_conn = torch.zeros((B, H, W)).to(tfn.device)
    fn_np = fn.cpu().detach().numpy() 
    tfn_np = tfn.cpu().detach().numpy()
    for i in range(B):
        fn = fn_np[i, :, :]
        tfn = tfn_np[i, :, :]
        ttfn = np.zeros_like(fn)
        mask_label, label_num = measure.label(fn, connectivity=1, background=0, return_num=True) 
        image_props = measure.regionprops(mask_label, cache=False) 
        for li in range(label_num):
            image_prop = image_props[li]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            tfn_sub = tfn[min_row:max_row, min_col:max_col].copy()

            tfn_sub[bool_sub < 1.0] = 0
            if np.sum(tfn_sub) > 0:
                ttfn[min_row:max_row, min_col:max_col] += bool_sub

        tfn_conn[i, :, :] = torch.from_numpy(ttfn).to(tfn_conn.device)
        tfn_conn[tfn_conn > 1] = 1
    return tfn_conn


def topo_preserve_conn_torch(label_ori, pred_ori, label_back_skelen=None, label_fore_skelen=None, is_show=False):
    label = label_ori  # B, H, W
    pred = pred_ori   # B, H, W 

    pred_back_skelen, pred_fore_skelen = calc_back_skelen_torch(pred)  # get skeletons for predictions
    if label_back_skelen is None:
        label_back_skelen, label_fore_skelen = calc_back_skelen_torch(label)  # get skeletons for train_labels, can be pre-loaded

    fn = minus_torch(label, pred)  # fn  train_labels = 1 & pred = 0
    fp = minus_torch(pred, label)  # fp  train_labels = 0 & pred = 1
    tn =  F.relu((1-label_ori) - fp)  # tn train_labels = 0 & pred = 0
    tp = F.relu(label_ori - fn)   # tp train_labels = 1 & pred = 1
    
    # get critical pixels
    def get_tfn(p, l, p_b_sk, l_f_sk, get_conn=False):
        fn = minus_torch(l, p)  # fn  train_labels = 1 & pred = 0
        tfn = minus_torch(fn, 1-p_b_sk)  # fn - pred_back_skelen
        if get_conn:
            tfn = get_tfn_conn_torch(fn, tfn)
        tfn = tfn * fn  # critical pixels 2

        dp = soft_dilate(p.unsqueeze(1)).squeeze(1)  # tolerate some offset
        tfn2 = minus_torch(l_f_sk, dp)  # label_fore_sklen - dilate_pred
        tfn2 = get_tfn_conn_torch(fn, tfn2)
        tfn2 = tfn2 * fn  # critical pixels 1

        tfn += tfn2  # critical pixels
        tfn[tfn > 1] = 1

        return tfn
    
    tfn = get_tfn(pred, label, pred_back_skelen, label_fore_skelen, get_conn=True)
    tfp = get_tfn(label, pred, label_back_skelen, pred_fore_skelen, get_conn=False)  # Swapping 'pred' and 'train_labels' would change calculating TFN to calculating TFP

    dilate_label = soft_dilate(label_ori.unsqueeze(1)).squeeze(1)  # tolerate some offset
    tfp[dilate_label > 0] = 0

    ffp = minus_torch(fp, tfp)  # ffp = fp - tfp
    ffn = minus_torch(fn, tfn)  # ffn = fn - tfn

    return tp, tn, tfn, ffn, tfp, ffp


class WeightMapBortLoss(nn.Module):
    """
    calculate weighted loss with weight maps with prob
    """

    def __init__(self, init_weight=10.0, epsilon=1.0, device='cuda'):
        super(WeightMapBortLoss, self).__init__()
        self.w0 = init_weight
        self.epsilon = epsilon
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def forward(self, pred, target, weight_maps, class_weight, label_skelen, eps = 1e-12, method='skeaw_dilate_step20_iter2_bort', step=50, epoch=1, d_iter=2):
        """
        target: The target map, LongTensor, unique(target) = [0 1]  1 - boudary/foreground  0 - object/background
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        class_weight: The class balance weight
        method：Select the type of loss function
        step: time to introduce bort
        epoch: current epoch
        d_iter: dilation iters
        # """
        # pred = pred.to(self.device)
        # target = target.to(self.device)
        weight_maps = weight_maps.to(self.device)
        class_weight = class_weight.to(self.device)
        label_skelen = label_skelen.to(self.device)

        # mask = target.float()
        logit = torch.softmax(pred, dim=1)
        label_skelen = label_skelen.permute(0, 3, 1, 2).contiguous()
        label_back_skelen = label_skelen[:, 0, :, :]
        label_fore_skelen = label_skelen[:, 1, :, :]

        # FFP, TFP, FFN, TFN calc
        # print('epoch ', epoch, ' step ', step, ' d_iter ', d_iter, ' lamda ', self.epsilon)
        if epoch > step:
            with torch.no_grad():
                pred_max = torch.argmax(logit, dim=1)
                mask_real = target.float().squeeze(1)
                tp, tn, tfn, ffn, tfp, ffp = topo_preserve_conn_torch(mask_real, pred_max.float(), label_back_skelen=label_back_skelen, label_fore_skelen=label_fore_skelen, is_show=False)

            bort_weight_sum = 0.0
            fn_prob = tp * (1 - logit[:, 1, :, :]) + tfn * (1 - logit[:, 1, :, :]) + ffn * (logit[:, 1, :, :])
            bort_weight_sum += ((tp + tfn + ffn).sum())
            fp_prob = tn * (1 - logit[:, 0, :, :]) + tfp * (1 - logit[:, 0, :, :]) + ffp * (logit[:, 0, :, :])
            bort_weight_sum += ((tn + tfp + ffp).sum())

        loss = 0
        weight_maps = weight_maps.permute(0, 3, 1, 2).contiguous().float()
        weight_sum = 0
        obj_temp = None

        for idx in [1, 0]:
            if idx == 0:  # object/background
                mask = 1 - target.float()  # B, C, W, H
            else:
                mask = target.float()

            wc = class_weight[:, idx, :].unsqueeze(-1)
            w0 = self.w0
                
            this_weight = weight_maps[:, idx, :, :] 
      
            weight_map = w0 * this_weight + wc * mask[:, 0, :, :]   # formula(3) & (4) in paper
                

            bck_weight = weight_maps[:, 0, :, :]*w0 + class_weight[:, 0, :].unsqueeze(-1)*(1-mask[:, 0, :, :])
            label_dilate = soft_dilate(mask)
            label_dilate = soft_dilate(label_dilate)
            label_dilate = label_dilate[:, 0, :, :]
                #  md in formula(2) in paper
            obj_temp = ((1-weight_maps[:, 0, :, :])*w0 + wc*mask[:, 0, :, :]) * label_dilate
            obj_temp[obj_temp <= bck_weight] = 0.0
            weight_map += (obj_temp * (1-mask[:, 0, :, :]))


            loss += -1 * weight_map * (torch.log(logit[:, idx, :, :]) + eps)
            weight_sum += weight_map.sum()
        
        loss_wp = loss.sum() / (weight_sum + eps)
        if 'bort' in method and epoch > step:
            bort_loss = self.epsilon * (fn_prob+fp_prob).sum() / bort_weight_sum
            loss_wp += bort_loss
        return loss_wp