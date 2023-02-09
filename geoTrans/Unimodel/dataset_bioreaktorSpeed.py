import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')
import torch
import os
import csv
import glob
from PIL import Image
from matplotlib import pyplot as plt
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import visdom
import itertools
import torch.nn.functional as f
from utils import Config as cfg
import cv2

class Bioreaktor_Detection(Dataset):

    def __init__(self, root, resize, mode, translation_x=0.25, translation_y=0.25):
        super(Bioreaktor_Detection, self).__init__()

        self.root = root
        self.resize = resize
        self.max_tx = translation_x
        self.max_ty = translation_y

        self.name2label = {} # "sq...":0 ... 1
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        self.transformation_dic = {}
        #  add GeoTrans Label 72
        for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4))):
            self.transformation_dic[i] = transform

        # save path
        self.images = []
        self.labels = []
        self.images_train, self.labels_train, self.images_testanormal, self.labels_testanormal, self.images_testanormal1, self.labels_testanormal1 = self.load_csv('train.csv', 'test_big.csv', 'test_small.csv')

        # traindata
        if mode == 'train':
            self.images = self.images_train[:cfg.NUM_TRANS*500]
            self.labels = self.labels_train[:cfg.NUM_TRANS*500]
        # testdata anormal
        elif mode == 'testbig':

            self.images = self.images_testanormal[:cfg.NUM_TRANS*200]
            self.labels = self.labels_testanormal[:cfg.NUM_TRANS*200]
        elif mode == 'testsmall':

            self.images = self.images_testanormal1[:int(cfg.NUM_TRANS*200)]
            self.labels = self.labels_testanormal1[:int(cfg.NUM_TRANS*200)]
        ## vali data normal nicht trainiert
        else:
            self.images = self.images_train[cfg.NUM_TRANS*4000:cfg.NUM_TRANS*4200]
            self.labels = self.labels_train[cfg.NUM_TRANS*4000:cfg.NUM_TRANS*4200]

    def load_csv(self, filename_train, filename_testanormal, filename_testanormal1):

        if not (os.path.exists(os.path.join(self.root, filename_train)) and os.path.exists(os.path.join(self.root, filename_testanormal)) and os.path.exists(os.path.join(self.root, filename_testanormal1))):
            train = []
            testanormal1 = []
            testanormal = []

            for name in self.name2label.keys():
                # 'cat\\1.jpg
                # print(os.path.join(self.root, name))
                if name == 'speed400':
                    train += glob.glob(os.path.join(self.root, name, '*.png'))
                    train += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    train += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif name == 'speedsmaller300':
                    testanormal1 += glob.glob(os.path.join(self.root, name, '*.png'))
                    testanormal1 += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    testanormal1 += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif name == 'speedbigger500':
                    testanormal += glob.glob(os.path.join(self.root, name, '*.png'))
                    testanormal += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    testanormal += glob.glob(os.path.join(self.root, name, '*.jpeg'))


            # 24946 ['E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL
            # \\Chapter9_1_CustomDataset\\Cat\\0.jpg',

            print(len(train), train[0])
            print(len(testanormal1), testanormal1[0])
            print(len(testanormal), testanormal[0])

            random.shuffle(train)
            random.shuffle(testanormal1)
            random.shuffle(testanormal)
            with open(os.path.join(self.root, filename_train), mode='w', newline='') as f1:
                writer_1 = csv.writer(f1)
                for img, i in itertools.product(train, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]
                    label = i
                    # 'cat', 0
                    writer_1.writerow([img, label])
                print('writen into csv file:', filename_train)

            with open(os.path.join(self.root, filename_testanormal), mode='w', newline='') as f2:
                writer_2 = csv.writer(f2)
                for img, i in itertools.product(testanormal, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]

                    label = i
                    # 'dog', 1
                    writer_2.writerow([img, label])
                print('writen into csv file:', filename_testanormal)
            with open(os.path.join(self.root, filename_testanormal1), mode='w', newline='') as f3:
                writer_3 = csv.writer(f3)
                for img, i in itertools.product(testanormal1, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]

                    label = i
                    # 'dog', 1
                    writer_3.writerow([img, label])
                print('writen into csv file:', filename_testanormal1)

        # read from csv file
        images_train, labels_train = [], []
        images_testanormal1, labels_testanormal1 = [], []
        images_testanormal, labels_testanormal = [], []
        with open(os.path.join(self.root, filename_train)) as f1:
            with open(os.path.join(self.root, filename_testanormal)) as f2:
                with open(os.path.join(self.root, filename_testanormal1)) as f3:
                    reader_1 = csv.reader(f1)
                    for row in reader_1:
                        # 'cat\\1.jpg', 0
                        img, label = row
                        label = int(label)

                        images_train.append(img)
                        labels_train.append(label)

                    reader_2 = csv.reader(f2)
                    for row in reader_2:
                        # 'dog\\1.jpg', 1
                        img, label = row

                        label = int(label)

                        images_testanormal.append(img)
                        labels_testanormal.append(label)

                    reader_3 = csv.reader(f3)
                    for row in reader_3:
                        # 'dog\\1.jpg', 1
                        img, label = row

                        label = int(label)

                        images_testanormal1.append(img)
                        labels_testanormal1.append(label)

        assert len(images_train) == len(labels_train)
        assert len(images_testanormal) == len(labels_testanormal)
        assert len(images_testanormal1) == len(labels_testanormal1)
        return images_train, labels_train, images_testanormal, labels_testanormal, images_testanormal1, labels_testanormal1

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        M = torch.zeros((2, 3))
        M[0, 0] = 1
        M[1, 1] = 1
        img, transformlabel = self.images[idx], self.labels[idx]
    # Add GeeoTrans
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('L'), # string path= > image data
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,],
                                 std=[0.229,])
        ])

        img = tf(img)
        if self.transformation_dic[transformlabel][0]:
            img = torch.flip(img, dims=[2])
        if self.transformation_dic[transformlabel][1] != 0 or self.transformation_dic[transformlabel][2] != 0:
            M[0, 2] = self.transformation_dic[transformlabel][1]
            M[1, 2] = self.transformation_dic[transformlabel][2]
            M = torch.unsqueeze(M, 0)
            grid = f.affine_grid(M, torch.unsqueeze(img, 0).size())
            img = f.grid_sample(input=torch.unsqueeze(img, 0), grid=grid, padding_mode='reflection', align_corners=True)
            img = torch.squeeze(img, 0)
        if self.transformation_dic[transformlabel][3] != 0:
            img = torch.rot90(img, k=self.transformation_dic[transformlabel][3], dims=(1, 2))

        transformlabel = torch.tensor(transformlabel)
        return img, transformlabel


    def denormalize(self, x_hat):

        mean = [0.485,]
        std = [0.229,]
        device = torch.device('cuda')
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x
