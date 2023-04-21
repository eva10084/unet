import os
import dataset

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import cv2
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image

class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的加载进来，做标签，总共2913张图片
        self.name = os.listdir(os.path.join(path, "SegmentationClass"))
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self, index):
        # 拿到的图片
        name = self.name[index]
        # 把标签名的格式改成jpg，与原始图片一致
        name2jpg = name[:-3] + "jpg"

        # 所有的原始图片和标签
        img_path = [os.path.join(self.path, i) for i in ("JPEGImages", "SegmentationClass")]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name2jpg))
        img_l = cv2.imread(os.path.join(img_path[1], name))
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, 256)
        img_l = self.__trans__(img_l, 256)

        return self.trans(img_o), self.trans(img_l)


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


# 训练器
class Trainer:

    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = UNet().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters())
        # 可以使用其他损失，比如DiceLoss、FocalLoss之类的
        self.loss_func = nn.BCELoss()
        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(Datasets(path), batch_size=4, shuffle=True, num_workers=4)

        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No Param!")
        os.makedirs(img_save_path, exist_ok=True)

    # 训练
    def train(self, stop_value):
        epoch = 1
        while True:
            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # 输出生成的图像
                out = self.net(inputs)
                loss = self.loss_func(out, labels)
                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, x_, y], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
                # print("image save successfully !")
            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")
            torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            # 备份
            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1


if __name__ == '__main__':
    t = Trainer(r"VOCdevkit\VOC2012", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./train_img')
    t.train(1)
