import torch
import json

from get_dataloader import get_train_test_loader, MyTrainDataLoaderIter, MyValDataLoaderIter
from Multi_Model_noGeo import WideResNet, ModelWrapper
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import mlflow
import itertools
import time
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from torch import nn
from matplotlib import pyplot as plt
from train_test import train_model, test_model
from metric import draw_roc, cf_matrix, get_f1, draw_pr

batchsz = 128
epochs = 15
out_num = 1
lr = 1e-4

# Hyperparameter wait for chosen
list_struc_cnn = [[10, 8],[10, 6]]
list_width_fcn = [20, 10, 5]
list_depth_fcn = [2, 1]
list_dropout = [0.1, 0.0]
# list_struc_cnn = [[10, 8]]
# list_width_fcn = [5]
# list_depth_fcn = [2, 1]
# list_dropout = [0.1]

torch.manual_seed(1234)
train_loader, test_loader = get_train_test_loader(batchsz, 64, 'Train50', 'Vali50')
_, test_loader100 = get_train_test_loader(batchsz, 64, 'Train100', 'Vali100')
_, test_loader200 = get_train_test_loader(batchsz, 64, 'Train200', 'Vali200')
_, test_loader300 = get_train_test_loader(batchsz, 64, 'Train300', 'Vali300')

device = torch.device('cuda')

all_conbination = list(itertools.product(list_struc_cnn, list_width_fcn, list_depth_fcn, list_dropout))
dict = {}
for i in range(3):
    loop = 0
    for struc_cnn, width_fcn, depth_fcn, dropout in all_conbination:
        mlflow.set_experiment("experiment lai_multimodel_trible")
        experiment = mlflow.get_experiment_by_name("experiment lai_multimodel_trible")
        name = "Multimodel_lai" + str(time.time())
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name = name):
            mlflow.log_param('loop', loop)
            model = WideResNet(struc_cnn[0], out_num, struc_cnn[1], width_fcn, 0, dropout, depth_fcn)
            params = {'cnn_depth': struc_cnn[0], 'cnn_width': struc_cnn[1], 'fcn_width': width_fcn, 'fcn_depth': depth_fcn, 'dropout': dropout}
            print(params)
            mlflow.log_params(params)
            # find lr
            trainloader_wrapper = MyTrainDataLoaderIter(train_loader)
            testloader_wrapper = MyValDataLoaderIter(test_loader)
            model_wrapper = ModelWrapper(model)
            model_wrapper = model_wrapper.to(device)
            optimizer = optim.Adam(model_wrapper.parameters(), lr=0.00005, weight_decay=5e-4)
            criterion = nn.BCEWithLogitsLoss()
            lr_finder = LRFinder(model_wrapper, optimizer, criterion, device="cuda")
            lr_finder.range_test(trainloader_wrapper, end_lr=0.1, num_iter=40, smooth_f=0.001)
            fig, ax = plt.subplots()
            # get best_lr
            Axes, bestlr = lr_finder.plot(skip_start=0, skip_end=0, ax=ax, suggest_lr=True)
            # ax can be further manipulated by the user after lr_finder.plot
            fig.savefig("bestLr.png")
            mlflow.log_param("best_lr", bestlr)
            lr_finder.reset()
            plt.clf()
            mlflow.log_artifact("bestLr.png")
            # train and test with best lr
            optimizer = optim.Adam(model.to(device).parameters(), lr=bestlr, weight_decay=5e-4)
            best_auc = 0
            best_f1 = 0
            global_step = 0
            for epoch in range(epochs):
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                global_step, avg_train_loss = train_model(model, device, train_loader, optimizer, epoch, epochs, global_step)
                mlflow.log_metric("avg_train_losses", avg_train_loss, step=epoch)
                anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader)
                mlflow.log_metric("val_acc_50", anormalacc, step=epoch)
                mlflow.log_metric("avg_val_loss_50", avg_val_loss, step=epoch)
                fpr, tpr, threshold = roc_curve(total_label, total_pred)
                auc_score = auc(fpr, tpr)
                mlflow.log_metric("auc_score_50", auc_score, step=epoch)
                precision, recall, threshold_pr = precision_recall_curve(total_label, total_pred)
                f1 = get_f1(precision, recall)
                mlflow.log_metric("f1_score_50", f1, step=epoch)

                if auc_score > best_auc:
                    best_auc = auc_score
                    mlflow.pytorch.log_model(model, 'model50', pip_requirements= ['torch==1.5.0', 'cloudpickle==2.2.0'])
                    best_threshold = draw_roc(total_label, total_pred, "roc50")
                    cf_matrix(total_pred, total_label, best_threshold, "cf50")

                    anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader100)
                    fpr, tpr, threshold = roc_curve(total_label, total_pred)
                    auc_score = auc(fpr, tpr)
                    mlflow.log_metric("auc_score_100", auc_score, step=epoch)
                    best_threshold = draw_roc(total_label, total_pred, "roc100")
                    cf_matrix(total_pred, total_label, best_threshold, "cf100")

                    anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader200)
                    fpr, tpr, threshold = roc_curve(total_label, total_pred)
                    auc_score = auc(fpr, tpr)
                    mlflow.log_metric("auc_score_200", auc_score, step=epoch)
                    best_threshold = draw_roc(total_label, total_pred, "roc200")
                    cf_matrix(total_pred, total_label, best_threshold, "cf200")

                    anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader300)
                    fpr, tpr, threshold = roc_curve(total_label, total_pred)
                    auc_score = auc(fpr, tpr)
                    mlflow.log_metric("auc_score_300", auc_score, step=epoch)
                    best_threshold = draw_roc(total_label, total_pred, "roc300")
                    cf_matrix(total_pred, total_label, best_threshold, "cf300")
                print(epoch, 'anormal_50 acc:', best_auc)
                print(loop)
            mlflow.log_param("best_auc50", best_auc)
            dict.setdefault(str(loop), []).append(best_auc)
            loop += 1
print(dict)
with open("sample.json", "w") as outfile:
    json.dump(dict, outfile)


# train_loader, test_loader = get_train_test_loader(batchsz, 64, 'Train100', 'Vali100')
# x1, x2, label = iter(test_loader).next()
# print('x1:', x1.shape, 'x2:', x2, 'label:', label)
# x1, x2, label = iter(test_loader).next()
# print('x1:', x1.shape, 'x2:', x2, 'label:', label)
# device = torch.device('cuda')
# model = WideResNet(16, 1, 8, 20, 0, 0, 2).to(device)
# print(model)
# torch.manual_seed(1234)
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# global_step = 0
# best_auc = 0

# for epoch in range(epochs):
#     avg_train_loss = train(model, device, train_loader, optimizer, epoch, epochs, global_step)
#     anormalacc, total_label, total_pred = test(model, device, test_loader)
#     fpr, tpr, threshold = roc_curve(total_label, total_pred)
#     auc_score = auc(fpr, tpr)
#     if auc_score > best_auc:
#         best_auc = auc_score
#         print(epoch, best_auc, threshold)

#     print(epoch, 'anormal acc:', anormalacc)
