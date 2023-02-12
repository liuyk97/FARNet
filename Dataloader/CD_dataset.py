import fnmatch
import os
import sys
import matplotlib.pyplot as plt
import tifffile
# from PIL.Image import Resampling, Transpose
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import math
import cv2
from utils.load_label import color2label
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = '5'


def get_params(preprocess, max_angle, crop_size, load_size, size, mode='train'):
    w, h = size
    new_h = h
    new_w = w
    angle = 0
    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size
    if 'rotate' in preprocess and mode == 'train':
        angle = random.uniform(0, max_angle)
        # print(angle)
        new_w = int(new_w * math.cos(angle * math.pi / 180) \
                    + new_h * math.sin(angle * math.pi / 180))
        new_h = int(new_h * math.cos(angle * math.pi / 180) \
                    + new_w * math.sin(angle * math.pi / 180))
        new_w = min(new_w, new_h)
        new_h = min(new_w, new_h)
    # print(new_h,new_w)
    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))
    # print('x,y: ',x,y)
    flip = None
    if mode == 'train':
        flip = random.random() > 0.5  # left-right
    return {'crop_pos': (x, y), 'flip': flip, 'angle': angle}


def get_transform(preprocess, crop_size, load_size, params=None, grayscale=False, method=InterpolationMode.BICUBIC,
                  convert=True, normalize=True, mode='train'):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    #  gaussian blur
    if 'blur' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __blur(img)))

    if 'rotate' in preprocess and mode == 'train':
        if params is None:
            transform_list.append(transforms.RandomRotation(5))
        else:
            degree = params['angle']
            transform_list.append(transforms.Lambda(lambda img: __rotate(img, degree)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'],
                                                                       crop_size)))

    if params is None:
        transform_list.append(transforms.RandomHorizontalFlip())
    elif params['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __blur(img):
    if img.mode == 'RGB':
        img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    return img


def __rotate(img, degree):
    if img.mode == 'RGB':
        # set img padding == 128
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        fff = Image.new('RGBA', rot.size, (128,) * 4)  # 灰色
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)
        return img
    else:
        # set label padding == 0
        img2 = img.convert('RGBA')
        rot = img2.rotate(degree, expand=1)
        # a white image same size as rotated image
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)
        img = out.convert(img.mode)
        return img


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    # print('imagesize:',ow,oh)
    # only 图像尺寸大于截取尺寸才截取，否则要padding
    if ow > tw and oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))

    size = [size, size]
    if img.mode == 'RGB':
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(img, (int((1 + size[1] - img.size[0]) / 2),
                              int((1 + size[0] - img.size[1]) / 2)))

        return new_image
    else:
        new_image = Image.new(img.mode, size, 255)
        # upper left corner
        new_image.paste(img, (int((1 + size[1] - img.size[0]) / 2),
                              int((1 + size[0] - img.size[1]) / 2)))
        return new_image


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class LEVID_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "dataset/LEVIR-CD_256x256/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.angle = 15
        self.crop_size = 256
        # self.pl_filename = 'pseudo_label_{}_from{}/'.format(model_name, date)
        self.path = os.listdir(self.data_path + self.mode + '/A/')
        # self.img_B = os.listdir(self.data_path + self.mode + '/B/')
        # self.lbl = os.listdir(self.data_path + self.mode + '/label/')
        # if self.mode == 'train':
        #     self.img_A = self.img_A[::8]
        #     self.img_B = self.img_B[::8]
        #     self.lbl = self.lbl[::8]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        # path = self.img_B[idx]
        # path = self.lbl[idx]
        img_A = Image.open(self.data_path + self.mode + '/A/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/B/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/label/' + path), dtype=np.uint32) / 255)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        A = transform(img_A)
        B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)
        L = transform_L(lbl)

        return {'A': A,
                'B': B,
                'L': L, 'path': path}


class SYSU_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/SYSU-CD/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.angle = 15
        self.crop_size = 256
        # self.pl_filename = 'pseudo_label_{}_from{}/'.format(model_name, date)
        self.path = os.listdir(self.data_path + self.mode + '/time1/')
        # self.lbl = os.listdir(self.data_path + self.mode + '/label/')
        # if self.mode == 'train':
        #     self.img_A = self.img_A[::8]
        #     self.img_B = self.img_B[::8]
        #     self.lbl = self.lbl[::8]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        # path_lbl = self.lbl[idx]
        img_A = Image.open(self.data_path + self.mode + '/time1/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/time2/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/label/' + path), dtype=np.uint32) / 255)

        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        img_A = transform(img_A)
        img_B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)
        lbl = transform_L(lbl)

        return {'A': img_A,
                'B': img_B,
                'L': lbl, 'path': path}


class DSIFN_set(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/DSIFN_256x256/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.angle = 15
        self.crop_size = 256
        self.path = os.listdir(self.data_path + self.mode + '/t1/')
        # self.lbl = os.listdir(self.data_path + self.mode + '/mask/')

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        path_lbl = path
        if self.mode == 'test':
            path_lbl = path_lbl.replace('jpg', 'tif')
        img_A = Image.open(self.data_path + self.mode + '/t1/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/t2/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/mask/' + path_lbl), dtype=np.uint32) / 255)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        A = transform(img_A)
        B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)
        L = transform_L(lbl)

        return {'A': A,
                'B': B,
                'L': L, 'path': path}


class CDD_set(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "dataset/CDD/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.angle = 15
        self.crop_size = 256
        self.path = os.listdir(self.data_path + self.mode + '/A/')
        # self.lbl = os.listdir(self.data_path + self.mode + '/OUT/')

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        # path_lbl = self.lbl[idx]
        img_A = Image.open(self.data_path + self.mode + '/A/' + path).convert('RGB')
        img_B = Image.open(self.data_path + self.mode + '/B/' + path).convert('RGB')
        lbl = Image.fromarray(
            np.array(Image.open(self.data_path + self.mode + '/OUT/' + path), dtype=np.uint32) / 255)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        img_A = transform(img_A)
        img_B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)
        lbl = transform_L(lbl)

        return {'A': img_A,
                'B': img_B,
                'L': lbl, 'path': path}


class WHU_CDset(Dataset):
    def __init__(self, mode="train"):
        self.data_path = "/data/sdu08_lyk/data/Building_change_detection_dataset_add_256x256/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.angle = 15
        self.crop_size = 256
        self.path = os.listdir(self.data_path + '/2012/whole_image/{}/image/'.format(self.mode))
        self.lbl = os.listdir(self.data_path + '/change_label/{}/'.format(self.mode))

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        path_2012 = self.path[idx]
        path_2016 = self.path[idx].replace('2012', '2016')
        path_change = self.path[idx].replace('2012_{}'.format(self.mode), 'change_label')
        img_A = Image.open(self.data_path + '/2012/whole_image/{}/image/'.format(self.mode) + path_2012).convert('RGB')
        img_B = Image.open(self.data_path + '/2016/whole_image/{}/image/'.format(self.mode) + path_2016).convert('RGB')
        lbl_A = tifffile.imread(self.data_path + '/2012/whole_image/{}/label/'.format(self.mode) + path_2012)
        lbl_B = tifffile.imread(self.data_path + '/2016/whole_image/{}/label/'.format(self.mode) + path_2016)
        lbl = tifffile.imread((self.data_path + '/change_label/{}/'.format(self.mode) + path_change))
        # lbl_A = np.where(lbl_A == False, 0, 255).astype(np.uint32)
        # lbl_B = np.where(lbl_B == False, 0, 255).astype(np.uint32)
        # lbl = np.where(lbl == False, 0, 255).astype(np.uint32)
        lbl_A = Image.fromarray(lbl_A / 255)
        lbl_B = Image.fromarray(lbl_B / 255)
        # lbl = Image.open(self.data_path + '/change_label/{}/'.format(self.mode) + path_lbl)
        lbl = Image.fromarray(lbl / 255)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        A = transform(img_A)
        B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)
        L_A = transform_L(lbl_A)
        L_B = transform_L(lbl_B)
        L = transform_L(lbl)

        return {'A': A,
                'B': B,
                'L_A': L_A, 'L_B': L_B, 'L': L}


class SECONDset(Dataset):  # 标签部分不对
    def __init__(self, mode="train", model_name='fcn'):
        self.data_path = "/data/sdu08_lyk/data/SECOND/"
        self.mode = mode
        self.preprocess = 'resize_and_crop'
        self.load_size = 512
        self.angle = 15
        self.crop_size = 512

        # self.images = os.listdir(self.data_path + 'im' + self.date)
        # self.labels = os.listdir(self.data_path + 'label' + self.date)
        # self.change_label = os.listdir(self.data_path + 'change_label/')
        # self.conf_file = [i.replace(".JPG", "_conf_up.npy") for i in self.images]

        if self.mode == 'train':
            filename_csv = open("/data/sdu08_lyk/data/SECOND/train.csv", "r")
        else:
            filename_csv = open("/data/sdu08_lyk/data/SECOND/val.csv", "r")
        reader = csv.reader(filename_csv)
        self.filename = []
        for item in reader:
            if reader.line_num == 1:
                continue
            self.filename.append(item[1])
        filename_csv.close()

    def __getitem__(self, idx):
        img_name = self.filename[idx].split('/')[-1]
        lbl_name = self.filename[idx].split('/')[-1]
        img_A = Image.open(self.data_path + 'im1/' + self.filename[idx]).convert('RGB')
        img_B = Image.open(self.data_path + 'im2/' + self.filename[idx]).convert('RGB')
        lbl_A = np.array(Image.open(self.data_path + 'label1/' + self.filename[idx]), dtype=np.uint32)
        lbl_B = np.array(Image.open(self.data_path + 'label2/' + self.filename[idx]), dtype=np.uint32)
        lbl_A = np.array(color2label(lbl_A, dataset='SECOND'), dtype=np.uint32)
        lbl_B = np.array(color2label(lbl_B, dataset='SECOND'), dtype=np.uint32)
        change_map = Image.fromarray(np.array(np.where(lbl_A != lbl_B, 1, 0), dtype=np.uint32))
        lbl_A = Image.fromarray(lbl_A)
        lbl_B = Image.fromarray(lbl_B)
        transform_params = get_params(self.preprocess, self.angle, self.crop_size, self.load_size, img_A.size,
                                      mode=self.mode)
        transform = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                  mode=self.mode)

        A = transform(img_A)
        B = transform(img_B)

        transform_L = get_transform(self.preprocess, self.crop_size, self.load_size, transform_params,
                                    method=InterpolationMode.NEAREST, normalize=False,
                                    mode=self.mode)

        L_A = transform_L(lbl_A)
        L_B = transform_L(lbl_B)
        change_map = transform_L(change_map)

        return {'A': A, 'paths_img': img_name,
                'B': B, 'paths_L': lbl_name,
                'L_A': L_A, 'L_B': L_B, 'change_map': change_map}

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    set = SYSU_CDset('test')
    loader = DataLoader(set, batch_size=1, num_workers=4, shuffle=True,
                        pin_memory=True)
    for idx, data in enumerate(loader):
        A = data['A'].squeeze().cpu().numpy().transpose(1, 2, 0)
        B = data['B'].squeeze().cpu().numpy().transpose(1, 2, 0)
        # L_A = data['L_A'].squeeze().cpu().numpy() * 255
        # L_B = data['L_B'].squeeze().cpu().numpy() * 255
        path = data['path']
        change_map = data['L'].squeeze().cpu().numpy() * 255
        # change_map_pred = np.where(L_A != L_B, 1, 0)
        plt.subplot(131)
        plt.axis('off')
        plt.imshow(A)
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(B)
        # plt.subplot(233)
        # plt.axis('off')
        # plt.imshow(L_A)
        # plt.subplot(234)
        # plt.axis('off')
        # plt.imshow(L_B)
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(change_map)
        # plt.subplot(144)
        # plt.axis('off')
        # plt.imshow(change_map_pred)
        plt.title(path)
        plt.show()
