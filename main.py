import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

import os

data_path = './train'
train_csv_path = './train.csv'

model_path = None

batch_size = 1
epochs = 100
lr = 1e-3
acc_grad_rate = 1  # сколько аккумулировать

backup_rate = 3  # шагов с учетом аккумуляции

data = pd.read_csv(train_csv_path)
train_path = data_path

train_pics_pathes = []

for cases in os.listdir(train_path):
    for days in os.listdir(os.path.join(train_path, cases)):
        for slices in os.listdir(os.path.join(train_path, cases, days, 'scans')):
            train_pics_pathes.append(os.path.join(cases, days, 'scans', slices))

id_pics_matching = dict()
for pics_path in train_pics_pathes:
    pics_path_splitted = pics_path.split('\\')
    name_splitted = pics_path_splitted[3].split('_')
    pics_id = pics_path_splitted[0] + '_' + pics_path_splitted[1].split('_')[1] + '_' + name_splitted[0] + '_' + name_splitted[1]
    id_pics_matching[pics_id] = os.path.join(train_path, pics_path)

data = data.dropna().reset_index()


def decode_rle(img, seq):
    img = img.clone()
    seq = seq.split()
    for start in range(0, len(seq), 2):
        start_x = int(seq[start]) % img.shape[2]
        start_y = int(seq[start]) // img.shape[1]
        for pix in range(start_x, start_x+int(seq[start+1])):
            img[0][start_y][pix] = 65536
    return img


def encode_line(img, i):
    pix = 0
    rle = []
    while pix < img.shape[2]:
        if img[0][i][pix] == 65536:
            start = pix
            while pix < img.shape[2] and img[0][i][pix] == 65536:
                pix += 1
            rle.append(str(i*img.shape[1] + start))
            rle.append(str(pix-start))
        pix += 1
    return rle


def encode_rle(img):
    rle = []
    for i in range(img.shape[1]):
        rle += encode_line(img, i)
    return ' '.join(rle)


data_full = dict()

for ids in data['id']:
    data_full[ids] = ['', '', '']

for i in range(len(data)):
    class_seg = data['class'][i]
    if class_seg == 'stomach':
        data_full[data['id'][i]][0] += data['segmentation'][i]
    if class_seg == 'small_bowel':
        data_full[data['id'][i]][1] += data['segmentation'][i]
    if class_seg == 'large_bowel':
        data_full[data['id'][i]][2] += data['segmentation'][i]


def move_to(data, device):
    """
    moving data to device
    :param data: data to move
    :param device: device
    :return: moved data
    """
    if isinstance(data, (list, tuple)):
        return [move_to(x, device) for x in data]
    return data.to(device, non_blocking=True)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Downsampler(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.pooling = pooling

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, X):
        X = self.act(self.conv1(X))
        X = self.act(self.conv2(X))
        X = self.act(self.conv3(X))
        if self.pooling:
            X = self.pool(X)
        return X


class Upsampler(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels // 2, 3)
        self.conv2 = torch.nn.Conv2d(in_channels // 2, in_channels // 2, 3)
        self.act = torch.nn.ReLU()

    def forward(self, X, X_cat):
        X = self.act(self.deconv(X))
        X = torch.cat([X, X_cat], axis=1)
        X = self.act(self.conv1(X))
        X = self.act(self.conv2(X))
        return X


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampler1 = Downsampler(1, 64)
        self.downsampler2 = Downsampler(64, 128)
        self.downsampler3 = Downsampler(128, 256)
        self.downsampler4 = Downsampler(256, 512)
        self.downsampler5 = Downsampler(512, 1024, pooling=False)

        self.upsampler1 = Upsampler(1024)
        self.upsampler2 = Upsampler(512)
        self.upsampler3 = Upsampler(256)
        self.upsampler4 = Upsampler(128)

        self.final_conv = torch.nn.Conv2d(64, 3, 3)  # 3 channels for seg. maps of stomach, large and small bowel
        self.final_act = torch.nn.Sigmoid()

    def copy_crop(self, X, shape):
        top = (X.shape[2] - shape) // 2  # as same as left
        return torchvision.transforms.functional.crop(X, top, top, shape, shape).clone()

    def forward(self, X):
        X = self.downsampler1(X)
        X_1 = X
        X = self.downsampler2(X)
        X_2 = X
        X = self.downsampler3(X)
        X_3 = X
        X = self.downsampler4(X)
        X_4 = X
        X = self.downsampler5(X)

        X = self.upsampler1(X, self.copy_crop(X_4, 48))
        X = self.upsampler2(X, self.copy_crop(X_3, 88))
        X = self.upsampler3(X, self.copy_crop(X_2, 168))
        X = self.upsampler4(X, self.copy_crop(X_1, 328))

        X = self.final_conv(X)
        return self.final_act(X)


def batch_loader(data, batch_size, id_pics_matching):
    ind = 0
    data_keys = list(data_full.keys())
    while ind + batch_size < len(data_keys) / 10000:
        X = torch.zeros([batch_size, 1, 572, 572])
        y = torch.zeros([batch_size, 3, 572, 572])

        for i in range(batch_size):
            X_new = torchvision.transforms.ToTensor()(Image.open(id_pics_matching[data_keys[ind + i]])).type(
                torch.float32) / (2 ** 16)
            X[i] = torchvision.transforms.Resize([572, 572])(X_new)

            for j in range(3):
                y_new = decode_rle(X_new, data_full[data_keys[ind + i]][j]).type(torch.float32) / (2 ** 16)
                y[i][j] = torchvision.transforms.Resize([572, 572])(y_new)
        ind += batch_size

        yield X, y


def batch_weights(y, n=1):
    res = torch.log(n*y+1)+0.2
    return res


torch.cuda.empty_cache()


model = move_to(UNet(), device)

if model_path:
    model.load_state_dict(torch.load(model_path))

opt = torch.optim.Adam(model.parameters(), lr=lr)

av_loss = 0

av_epoch_loss = 0
epoch_steps = 0

n_iters_loss = 1  # как часто выводим

cnt = 0
steps = 0

for epoch in range(epochs):
    av_epoch_loss = 0
    epoch_steps = 0
    for X, y in batch_loader(data, batch_size, id_pics_matching):

        X, y = move_to(X, device), move_to(y, device)
        out = torchvision.transforms.Resize([572, 572])(model(X))

        loss = torch.nn.BCELoss(weight=move_to(batch_weights(y, 1), device))(out, y) / acc_grad_rate
        loss.backward()

        av_loss += loss.detach().cpu().numpy()

        av_epoch_loss += loss.detach().cpu().numpy() * acc_grad_rate
        epoch_steps += 1

        if cnt > 0 and cnt % acc_grad_rate == 0:
            opt.step()
            opt.zero_grad()
            steps += 1

            if steps % n_iters_loss == 0 and steps > 0:
                print(steps, av_loss / n_iters_loss)
                av_loss = 0

            if steps % backup_rate == 0 and steps > 0:
                print(f'(!) Backup {steps//backup_rate}')
                torch.save(model, f'models/backup_{steps//backup_rate}.pth')

        cnt += 1
        torch.cuda.empty_cache()
    print(f'Epoch {epoch} finished with av. loss: {av_epoch_loss/epoch_steps}')

torch.save(model, 'models/model.pth')
print('Learning finished successfully!')
