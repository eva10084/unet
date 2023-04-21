import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
#import nibabel as nib
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)


EPOCH = 30  # 轮数
KLDLamda = 1.0   # Kullback-Leibler散度的权重

PredLamda=1e3
DisLamda=1e-4
LR = 1e-3   # 代表Adam优化器的初始学习率
ADA_DisLR = 1e-4  # 代表判别器的学习率

WEIGHT_DECAY =1e-5   # 代表Adam优化器的权重衰减系数
WORKERSNUM = 0   # 代表用于数据加载的进程数  PS 初始为10，只有0时可以运行
prefix = 'experiments/model'   # 返回上一级目录，代表实验结果保存的路径
# prefix = 'gdrive/MyDrive/vae/experiments/loss_tSNE'  # Google云盘
dataset_dir = 'Dataset/small_Patch192'  # 返回上一级目录，代表数据集所在的路径
# dataset_dir = 'Dataset/Patch192'  # 返回上一级目录，代表数据集所在的路径
source = 'C0'
target = 'LGE'

ValiDir = dataset_dir +'/'+target+'_Vali/'  # 代表验证集数据所在的路径

BatchSize = 5  # 代表每个批次的样本数
KERNEL = 4   # 代表卷积核的大小

# 将CUDA_VISIBLE_DEVICES环境变量设置为'0'，以使用第一个GPU来运行TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


evaluation_interval = 10  # 指定模型在训练期间进行评估的频率
save_interval = 300     # 指定模型在训练期间保存的频率
num_cls = 4     # 数据集中类别的数量
keep_rate_value = 0.75   # 可能用于定义训练期间的dropout或正则化，其中keep_rate_value指示在dropout层中保留单元的概率
is_training_value = True   # 指示模型是正在训练还是正在评估的布尔值

if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda")  # GPU 可用
else:
    print("CPU")
    device = torch.device("cpu")   # 只能使用 CPU


def main():
    SAVE_DIR=prefix+'/haha'   # 保存参数路径

    model = UNet()
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 调整学习率
    # criterion = nn.CrossEntropyLoss()

    SourceData = source_TrainSet(dataset_dir)
    dataloader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,
                                   pin_memory=True, drop_last=True)


    if not os.path.exists(SAVE_DIR):   # 如果保存训练结果的目录不存在，则创建该目录
        os.mkdir(SAVE_DIR)


    for epoch in range(EPOCH):
        # 设置为训练模式
        model.train()

        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)
            # label_onehot = torch.FloatTensor(label.size(0), 4, label.size(1), label.size(2)).to(device)
            # label_onehot.zero_()
            # label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

            label = label.to(device, dtype=torch.int64).squeeze()  # 需要int64参与运算，squeeze：(n,1,512,512)->(n,512,512)
            label[label > 0] = 1  # !=0的地方即为1
            out = model(image)
            loss = nn.BCELoss()(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\nEpoch: {epoch}/{EPOCH}, Loss: {loss}")
            if epoch % 1 == 0:
                torch.save(model.state_dict(), 'res.pkl')





if __name__ == '__main__':
    main()


