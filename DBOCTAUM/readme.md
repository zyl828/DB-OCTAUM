# OCTAUM

## 简介
该项目旨在实现血管分割任务，采用OCTAUM算法，使用OCTA-SS,OCTA500（OCTA-3M与OCTA-6M）这两个数据集，前期也用过ROSE数据集但后来弃用，OCTA500数据集和ROSE数据集是半公开数据集，使用前需要在官网上发邮件获得许可。

## 环境配置
在开始之前，请确保你的环境中安装了以下依赖项：

### 必要依赖
- Ubuntu 20.04
- CUDA 11.8

### 安装步骤
1. Create a virtual environment: `conda create -n umamba python=3.10 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
5. `cd U-Mamba/umamba` and run `pip install -e .`
6. 隐藏层安装（可选） pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details

## 数据预处理

### 数据集预处理
1.nnUNetV2对于数据集的原图只接受三通道png图，标签只接受单通道png图，需将tif等图像格式转化为png图后，再堆叠，堆叠方式参考（当然也可自定义堆叠方式）https://blog.csdn.net/qq_48951688/article/details/124786821
2.标签图在输入前需要进行0 1二值化，参考https://blog.csdn.net/sdhdsf132452/article/details/124155656

### 数据扩增
OCTA-SS的数据量过小，直接输入会出现类似欠拟合的情况，需要先进行数据扩增，可参考https://blog.csdn.net/weixin_45912366/article/details/127855494
（本来也使用过ROSE数据集，但ROSE系列数据集不扩增也会出现欠拟合现象，扩增后灰度值又会发生改变导致二值化受阻，强行二值化后DICE最高也只有0.75左右，遂放弃，可以尝试别的扩增方式）

### 配置自定义数据集
创建nnUNet_raw 保存格式转换后的数据集，创建nnUNet_result 保存结果文件，修改文件nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py使其与自定义数据集对应后并运行。这一部分参考https://blog.csdn.net/qq_44776065/article/details/131048099?ops_request_misc=&request_id=&biz_id=102&utm_term=nnunetv2%E8%AE%AD%E7%BB%83%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-131048099.142 中的2.3

### 配置环境变量
```vim .bashrc```
在最后添加

```
export nnUNet_raw="/sharefiles1/hanliqiang/GitCode/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/sharefiles1/hanliqiang/GitCode/nnUNet/nnunetv2/preprocessing"
export nnUNet_results="/sharefiles1/hanliqiang/GitCode/nnUNet/nnUnet_results"
```
然后
```source .bashrc```

### 格式预处理
在数据集处理好后（注意原图和标签的ID要一一对应）
```nnUNetv2_plan_and_preprocess -d DATASET_ID（即为自选ID） --verify_dataset_integrit```

## 训练
```nnUNetv2_train 110 2d 9  -tr nnUNetTrainerUMambaEncNoAMP```
110即为你的数据集ID，9代表十折交叉验证，可更改验证方式，在模型主函数里修改do_split函数后重新划分数据集即可，默认是五折交叉，nnUNetTrainerUMambaEncNoAMP也可替换成别的网络模型，nnUNetTrainerUMambaEnc存在bug，不要选。

## 推理

### 寻找最佳配置
位置： nnunetv2/evaluation/find_best_configuration.py
更改主函数中的数据集ID为你的自定义ID，交叉验证改为你选择的折数，然后运行

### 推理
```nnUNetv2_predict -i /root/autodl-tmp/U-Mamba-main/data/nnUNet_raw/Dataset110_OCTASegmentation/imagesTs -o /root/autodl-tmp/U-Mamba-main/data/nnUNet_predict_result/Dataset110_result -d 110 -c 2d -f 9 -tr nnUNetTrainerUMambaEncNoAMP --disable_tta```
/root/autodl-tmp/U-Mamba-main/data/nnUNet_raw/Dataset110_OCTASegmentation/imagesTs即为进行推理的原图，/root/autodl-tmp/U-Mamba-main/data/nnUNet_predict_result/Dataset110_result为推理结果储存的位置

### 后处理
```nnUNetv2_apply_postprocessing -i OUTPUT_FOLDER -o OUTPUT_FOLDER_PP -pp_pkl_file /sharefiles1/hanliqiang/GitCode/nnUNet/nnUnet_results/Dataset110_StentSegmentation/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_5/postprocessing.pkl -np 8 -plans_json /sharefiles1/hanliqiang/GitCode/nnUNet/nnUnet_results/Dataset110_StentSegmentation/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_5/plans.json```
对于OCTAUM采用的数据集后处理貌似不起作用，此处不赘述。

## 评估
借用nnUNetV2的源码文件进行评估，位置：nnunetv2/evaluation/evaluate_predictions.py
对主函数进行修改，floder_ref改为测试集的GT文件
floder_pred 改

output_file 改为输出评价结果的json文件
labels_to_list_of_regions([1]) 前景是1，指定需要评估的区域为label1
文件后缀修改为.png文件
修改完成后运行即可，可得到DICE与Iou两个指标。

## 损失函数
使用clDice函数作为损失函数后，因为没有对应修改损失函数曲线绘制部分的代码，所以损失曲线会失真，但不影响结果。





