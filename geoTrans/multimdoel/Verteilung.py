import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

from dataset_bioreaktorMulti import Bioreaktor_Detection
from torch.utils.data import DataLoader, Dataset
from geoTrans.multimdoel.Multi_Model import WideResNet as Multimodel
import torch
import torch.optim as optim
from torch import nn
import visdom
from utils import Config as cfg
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np

model = Multimodel(10, 72, 6)
root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll"
model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiAll3lossRes_106_1.mdl'))


# print model arch
print(model)

# print model parameters
layer_list = []
params_list = []
for name, param in model.named_parameters():
    if param.requires_grad:
#         print(name, param.data.type())
        layer_list.append(name)
        params_list += param.data.view(-1, 1)

params_list = [round(float(i.numpy().tolist()[0]), 3) for i in params_list]
nonzero_norm_list=[i for i in params_list if i!=0]
zero_num = 0
for k in params_list:
    if k == 0:
        zero_num += 1

print("all parameters: ", len(params_list))
print("all non zero parameters: ", len(nonzero_norm_list))
print("max value: ", max(params_list))
print("min value: ", min(params_list))
print("zero count: ", zero_num)
# print("top 10", params_list[:10])
plt.hist(params_list[:], bins=300)
plt.title("parameter value range")
plt.xlabel("Value Range")
plt.ylabel("Counts")
plt.show()
