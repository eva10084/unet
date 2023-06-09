import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn
from torch import optim
from tool import *
from unet import UNet
from torchvision.utils import save_image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)

EPOCH = 20  # 轮数
KLDLamda = 1.0  # Kullback-Leibler散度的权重

PredLamda = 1e3
DisLamda = 1e-4
LR = 1e-3  # 代表Adam优化器的初始学习率
ADA_DisLR = 1e-4  # 代表判别器的学习率

WEIGHT_DECAY = 1e-5  # 代表Adam优化器的权重衰减系数
WORKERSNUM = 0  # 代表用于数据加载的进程数  PS 初始为10，只有0时可以运行
prefix = 'experiments/model'  # 返回上一级目录，代表实验结果保存的路径
# prefix = 'gdrive/MyDrive/vae/experiments/loss_tSNE'  # Google云盘
dataset_dir = 'Dataset/small_Patch192'  # 返回上一级目录，代表数据集所在的路径
# dataset_dir = 'Dataset/Patch192'  # 返回上一级目录，代表数据集所在的路径
source = 'C0'
target = 'LGE'

ValiDir = dataset_dir + '/' + target + '_Vali/'  # 代表验证集数据所在的路径

BatchSize = 5  # 代表每个批次的样本数
KERNEL = 4  # 代表卷积核的大小

# 将CUDA_VISIBLE_DEVICES环境变量设置为'0'，以使用第一个GPU来运行TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

evaluation_interval = 10  # 指定模型在训练期间进行评估的频率
save_interval = 300  # 指定模型在训练期间保存的频率
num_cls = 4  # 数据集中类别的数量
keep_rate_value = 0.75  # 可能用于定义训练期间的dropout或正则化，其中keep_rate_value指示在dropout层中保留单元的概率
is_training_value = True  # 指示模型是正在训练还是正在评估的布尔值

if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda")  # GPU 可用
else:
    print("CPU")
    device = torch.device("cpu")  # 只能使用 CPU


def one_hot(label):
    label_onehot = torch.nn.functional.one_hot(label, num_classes=4)
    label_onehot = torch.squeeze(label_onehot, dim=1).permute(0, 3, 1, 2)
    # label_onehot=torch.nn.functional.one_hot(label, 4, dim=1).permute(0, 3, 1, 2)
    # label_onehot = torch.FloatTensor(label.size(0), 4, label.size(2), label.size(3)).zero_().to("cpu")
    # # 根据标签值对每个通道进行赋值
    # label_0 = (label == 0).nonzero()
    # label_onehot[:, 0, label_0[:, 2], label_0[:, 3]] = 1
    # label_85 = (label == 85).nonzero()
    # label_onehot[:, 1, label_85[:, 2], label_85[:, 3]] = 1
    # label_170 = (label == 170).nonzero()
    # label_onehot[:, 2, label_170[:, 2], label_170[:, 3]] = 1
    # label_255 = (label == 255).nonzero()
    # label_onehot[:, 3, label_255[:, 2], label_255[:, 3]] = 1
    return label_onehot.to(device)


def main():
    SAVE_DIR = prefix + '/haha'  # 保存参数路径

    model = UNet()
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 调整学习率
    # criterion = nn.CrossEntropyLoss()

    SourceData = source_TrainSet(dataset_dir)
    dataloader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,
                            pin_memory=True, drop_last=True)

    if not os.path.exists(SAVE_DIR):  # 如果保存训练结果的目录不存在，则创建该目录
        os.mkdir(SAVE_DIR)

    for epoch in range(EPOCH):
        # 设置为训练模式
        model.train()

        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device, dtype=torch.int64)  # 需要int64参与运算
            label_onehot = one_hot(label)
            # [print(i) for i in label_onehot[0][0]]
            print(label_onehot.shape)
            # exit()
            out = model(image)
            # _, out = torch.max(out, dim=1, keepdim=True)   # 将概率变成索引

            print(image.shape, label_onehot.shape, out.shape)
            loss = nn.BCELoss()(out.float(), label_onehot.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\nEpoch: {epoch}/{EPOCH}, Loss: {loss}")
            if epoch % 1 == 0:
                torch.save(model.state_dict(), 'res.pkl')


'''
            x = image[0]
            x_ = out[0]
            y = label_onehot[0]
            print(x.shape,x_.shape,y.shape)
            img = torch.stack([x, x_, y], 0)
            save_image(img.cpu(), "kk.png")
'''

if __name__ == '__main__':
    main()