import mlflow
import torch
from get_dataloader import get_train_test_loader
from Multi_Model_noGeo import WideResNet
from matplotlib import pyplot as plt
from train_test import test_model
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from metric import draw_roc, cf_matrix, get_f1, draw_pr
import csv
import os
import json

batchsz = 128

mlflow.set_tracking_uri('http://localhost:5001')
path = "models:/Multimodel_Best/1"
model1 = mlflow.pytorch.load_model(path)
path = "models:/Multimodel_Best/2"
model2 = mlflow.pytorch.load_model(path)
path = "models:/Multimodel_Best/3"
model3 = mlflow.pytorch.load_model(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader50 = get_train_test_loader(batchsz, 64, 'Train50', 'Vali50')


anormalacc, total_label, total_pred, avg_val_loss = test_model(model1, device, test_loader50)
fpr501, tpr501, threshold = roc_curve(total_label, total_pred)

best_threshold = draw_roc(total_label, total_pred, "roc501")
false1 = cf_matrix(total_pred, total_label, best_threshold, "cf501")
path =  '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll/Vali50'
root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/unimodelspeed/speedbigger500'
false_speed = []
false_Luft = []
speed = []
Luft = []
for i in false1:
    if i % 2 == 1:

        with open(path, 'r') as f:
            reader = csv.reader(f)
            result = list(reader)
            anormal_speed = int(result[(i)][2])
            anormal_luft = int(result[(i)][3])
            normal_speed = int(result[(i)-1][2])
            normal_luft = int(result[(i)-1][3])
            false_speed.append(abs(normal_speed-anormal_speed))
            false_Luft.append(abs(normal_luft-anormal_luft))

plt.hist(false_speed, bins=20)
plt.title('Distribution of FN (rotate speed)', fontsize=15)
plt.xlabel("Value Range", fontsize=15)
plt.ylabel("Counts", fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.savefig("Verteilung"+"roc501"+'.png')
plt.show()


plt.hist(false_Luft, bins=20)
plt.title('Distribution of FN (volume strom)', fontsize=15)
plt.xlabel("Value Range", fontsize=15)
plt.ylabel("Counts", fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.savefig("Verteilung"+"roc501"+'.png')
plt.show()


for i in range(0,1000):
    with open(path, 'r') as f2:
        reader = csv.reader(f2)
        result = list(reader)
        normal_speed = int(result[(i)*2][2])
        normal_luft = int(result[(i)*2][3])
        anormal_speed = int(result[(i)*2+1][2])
        anormal_luft = int(result[(i)*2+1][3])
        speed.append(abs(normal_speed-anormal_speed))
        Luft.append(abs(normal_luft-anormal_luft))

plt.hist(speed, bins=20)
plt.title('Distribution of Rotate Speed Difference', fontsize=15)
plt.xlabel("Value Range", fontsize=15)
plt.ylabel("Counts", fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.savefig("Verteilung"+"roc501"+'.png')
plt.show()


plt.hist(Luft, bins=20)
plt.title('Distribution of Volume Strom Difference', fontsize=15)
plt.xlabel("Value Range", fontsize=15)
plt.ylabel("Counts", fontsize=15)
plt.tick_params(axis='both', labelsize=15)
plt.savefig("Verteilung"+"roc501"+'.png')
plt.show()
