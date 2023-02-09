import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from Model import WideResNet
import random

def draw_CAM(device, model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    draw Class Activation Map
    :param model: load Pytorch model with Weight
    :param img_path: path
    :param save_path: save path for result
    :param transform:
    :param visual_heatmap:
    :return:
    '''
    # load image and transform
    img = Image.open(img_path).convert('L')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
    img=img
    # get feature/score
    model.eval()
    output, features = model(img)


    # Help Function
    def extract(g):
        global features_grad
        features_grad = g

    # get highest score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward() #

    grads = features_grad   #

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # batch size=1，squeeze dim 0（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512 filter for last layer
    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]

    #
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # get heatmap
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

img_root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData2/speedbigger500'
save_root= '/home/lai/Downloads/heamapSpeed/TN3.png'
device = torch.device('cuda')
tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,],
                                 std=[0.229,])
        ])

# ([ 17,  26,  28,  36,  37,  39,  58,  60,  69,  77,  86,  93, 100, 118,
#         132, 137, 149, 161, 170, 182, 191, 217, 303, 315, 316, 346, 355, 390])    normal 200 anormal
model = WideResNet(16, 72, 8)
print(model)
model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/modelspeed21681.mdl'))
# name_all = os.listdir(img_root)
# name_sample = random.sample(name_all, 100)
# i=0
# for name in name_sample:
#     img_path = os.path.join(img_root, name)
#     print(img_path)
#     save_path = os.path.join(save_root, '500_'+str(i)+'.png')
#     print(save_path)
#     draw_CAM(device, model,img_path,save_path,tf)
#     print(i)
#     i += 1
draw_CAM(device, model,'/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData2/speed400/1668783825383_camera_frame.png',save_root,tf)
