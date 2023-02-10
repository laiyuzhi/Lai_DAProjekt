import os, glob
import random
import csv
import cv2
from PIL import Image
import numpy as np


def load_csv(filename, root):

    if not os.path.exists(os.path.join(root, filename)):
        images = []

        images += glob.glob(os.path.join(root, "*.png"))
        images += glob.glob(os.path.join(root, "*.jpg"))
        images += glob.glob(os.path.join(root, "*.jpeg"))

        #     # 5189,  'E:\\data_lai\\unimodel_data\\Train_normal\\1668789435.312676_0.png',
        print(len(images))

        random.shuffle(images)
        with open(os.path.join(root, filename), mode="w", newline="") as f:
            writer = csv.writer(f)
            for img in images:
                # name = img.split(os.sep)[-2]
                writer.writerow([img])
            print("writen into csv file:", filename)

    images = []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img = row
            images.append(img)

    return images


class UserCrop:
    def __init__(self, filename, root, path):
        
        self.path = path
        self.root = root
        self.filename = filename

    def cut(self):
        img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
        if not os.path.exists(os.path.join(self.root, self.filename)):
            cv2.namedWindow("ROI selector", cv2.WINDOW_NORMAL)
            bbox = cv2.selectROI(img, False)
            with open(os.path.join(self.root, self.filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(bbox)
                print("writen into csv file:", self.filename)

        with open(os.path.join(self.root, self.filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                bbox = [int(x) for x in row]
                print(bbox)

        return img, bbox
     

def cut_picture(point, rawimages_path, image_path):
    img, bbox = UserCrop(point, rawimages_path, image_path).cut()
    cut = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    cv2.namedWindow("cut")
    cv2.imshow("cut", cut)
    cv2.waitKey(0)
   
def preprocess(root, save_root, point, rawimages_path, image_path):
   
    _, bbox = UserCrop(point, rawimages_path, image_path).cut()
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        print(os.path.join(root, name))
        i = 0
        for img in os.listdir(os.path.join(root, name)):
            if os.path.join(os.path.join(root, name), img).endswith('.png'):
                img_path = os.path.join(os.path.join(root, name), img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cut = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                if i <= 3999:
                    save_path = os.path.join(save_root, 'Train', name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img_path = os.path.join(save_path, img)
                else:
                    save_path = os.path.join(save_root, 'Test', name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img_path = os.path.join(save_path, img)
                cv2.imwrite(img_path, cut, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                i += 1
        print(i)

if __name__ =='__main__':
    name = 'bboxMultiSpeed'
    rawimages_path= '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultimodelSpeedLuft'
    image_path = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultimodelSpeedLuft/Speed400Luft2/1668792765397_camera_frame.png"
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultimodelSpeedLuft'
    save_root= '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultimodelSpeedLuft'
    cut_picture(name, rawimages_path, image_path)
    preprocess(root, save_root, name, rawimages_path, image_path)
