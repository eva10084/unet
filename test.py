import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import glob
from tool import *
from unet import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)

TestDir = 'Dataset/small_Patch192/C0/'
model_dir = 'res.pkl'
# model_dir = 'gdrive/MyDrive/vae/experiments/loss_tSNE/save_param0.001/best_model'  # Google云盘

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU 可用
else:
    device = torch.device("cpu")  # 只能使用 CPU


# slice = 12

def index_to_image(image):
  image[image == 0] = 0
  image[image == 1] = 85
  image[image == 2] = 160
  image[image == 3] = 255
  return image

def SegNet(dir, SegNet, gate):
    name = glob.glob(dir + '*C0.nii*')
    slice = 4
    SegNet.eval()

    for i in range(len(name)):
        itkimg = sitk.ReadImage(name[i].replace('\\', '/'))
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)
        print(npimg.shape)
        print(npimg[slice, :, :].shape)
        plt.imshow(npimg[slice, :, :], cmap='gray')
        plt.show()
        plt.savefig('init.png')

        data = torch.from_numpy(np.expand_dims(npimg, axis=1)).type(dtype=torch.FloatTensor).to(device)
        result = np.zeros((data.size(0), data.size(2), data.size(3)))

        # 对每个切片进行操作
        for slice in range(data.size(0)):
            output = SegNet(data[slice:slice + 1, :, :, :])
            # [print(i) for i in output[0,:,55,55]]
            # print(output.shape)
            truemax, result0 = torch.max(output, 1, keepdim=False)
            # print(result0.shape)
            # [print(i) for i in result0[:,55,55]]
            result0 = index_to_image(result0)
            # [print(i) for i in result0[:,55,55]]
            result0 = result0.detach().cpu().numpy()
            result[slice:slice + 1, :, :] = result0

        print(result.shape)
        print(result[slice, :, :].shape)
        plt.imshow(result[slice, :, :], cmap='gray')
        plt.show()
        plt.savefig('result.png')


def show(dir):
    name = glob.glob(dir + '*_manual.nii*')
    slice = 4

    for i in range(len(name)):
        itkimg = sitk.ReadImage(name[i].replace('\\', '/'))
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)
        print(npimg.shape)
        print(npimg[slice, :, :].shape)
        plt.imshow(npimg[slice, :, :], cmap='gray')
        plt.show()
        plt.savefig('real.png')


if __name__ == '__main__':
    show(TestDir)

    model = UNet()
    model.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))
    SegNet(TestDir, model.to(device), 0)
