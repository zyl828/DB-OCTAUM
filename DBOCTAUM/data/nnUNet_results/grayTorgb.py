import cv2
import numpy as np
from PIL import Image
import os

# 设置文件路径
file_path = 'D:/firefoxDownload/OCTAUM/OCTAUM_LastExp/OCT_6M_LS/UNTER/Dataset120_unetr_result/'
# image_file = os.path.join(file_path, 'label (29).png')  # 完整的图像文件路径/
image_file = os.path.join(file_path, 'label (1).png')

# 检查文件是否存在
if not os.path.exists(image_file):
    print(f"Error: File does not exist at path {image_file}")
else:
    # 尝试读取图像
    img_cv = cv2.imread(image_file)  # 这里不指定cv2.IMREAD_GRAYSCALE，因为后面要转换
    # 检查图像是否成功加载
    if img_cv is None:
        print(f"Error: Failed to load image at path {image_file}")
    else:
        img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # 如果是彩色图像，则转换为灰度
        img_pil = Image.fromarray(img_cv_gray)
        img_array = np.array(img_pil)

        img_array[img_array != 0] = 255
        img_binary = Image.fromarray(img_array.astype('uint8'))  # 确保类型正确
        # img_binary.show()  # 显示二值化后的图像（可选）

        # img_binary_path = os.path.join(file_path, 'labelnew (3).png')
        img_binary_path = os.path.join(file_path, 'labelnew (1).png')
        img_binary.save(img_binary_path)
