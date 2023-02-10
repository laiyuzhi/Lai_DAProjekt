import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataset_bioreaktorSpeed import Bioreaktor_Detection as Speed
from geoTrans.Unimodel.dataset_bioreaktorLuft import Bioreaktor_Detection as Luft
from torch.utils.data import DataLoader, Dataset
from Model import WideResNet
import torch
import torch.optim as optim
from torch import nn
import visdom
from utils import Config as cfg
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def load_data(dataname):

    if dataname == 'Speed300' or dataname == 'Speed500':
        print(dataname)

        root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData2'
        batchsz = cfg.BATCH_SIZE
        num_trans = cfg.NUM_TRANS
        if dataname == 'Speed500':
            test_db = Speed(root, 64, mode='testbig')
        if dataname == 'Speed300':
            test_db = Speed(root, 64, mode='testsmall')
        vali_db = Speed(root, 64, mode='vali')
        vali_loader = DataLoader(vali_db, batch_size=batchsz, num_workers=0)
        test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
        x, label = iter(vali_loader).next()
        print('x:', x.shape, 'label:', label.shape)

        device = torch.device('cuda')
        # viz = visdom.Visdom()
        model = WideResNet(16, num_trans, 8).to(device)
        model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/modelspeed21681.mdl'))

        # Eva for normal data
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for batchidx, (x, label) in pbar:
                x, label = x.to(device), label.to(device)

                # [b, 72]
                logits,_ = model(x)
                # [b]
                pred =  nn.functional.softmax(logits, dim=1)
                p = torch.tensor([pred[i, int(label[i].item())] for i in range(len(label))])
                # [b] vs [b] => scalar tensor
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = p
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, p), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TPFN = total_pred.float().sum(1)
            TPFN_prob = TPFN / cfg.NUM_TRANS

        # Eva for anormal data
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batchidx, (x, label) in pbar:
                x, label = x.to(device), label.to(device)

                # [b, 72]
                logits,_ = model(x)
                # [b]
                pred =  nn.functional.softmax(logits, dim=1)
                p = torch.tensor([pred[i, int(label[i].item())] for i in range(len(label))])

                # [b] vs [b] => scalar tensor
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = p
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, p), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TNFP = total_pred.float().sum(1)
            TNFP_prob = TNFP / cfg.NUM_TRANS

        TPFNTNFP_prob = torch.cat((TPFN_prob, TNFP_prob), 0)
        TPFNTNFP_label = torch.cat((torch.ones_like(TPFN), torch.zeros_like(TNFP)), 0)

        TPFNTNFP_label = torch.Tensor.cpu(TPFNTNFP_label)
        TPFNTNFP_prob = torch.Tensor.cpu(TPFNTNFP_prob)

    elif dataname == 'Luft':
        root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelLuft_data2'
        batchsz = cfg.BATCH_SIZE
        num_trans = cfg.NUM_TRANS
        test_db = Luft(root, 64, mode='test')
        vali_db = Luft(root, 64, mode='vali')
        vali_loader = DataLoader(vali_db, batch_size=batchsz, num_workers=0)
        test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
        x, label = iter(vali_loader).next()
        print('x:', x.shape, 'label:', label.shape)

        device = torch.device('cuda')
        # viz = visdom.Visdom()
        model = WideResNet(16, num_trans, 8).to(device)
        model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans/mdl/modelspeedfordata2.mdl'))

        # Eva for normal data
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
            for batchidx, (x, label) in pbar:
                x, label = x.to(device), label.to(device)

                # [b, 72]
                logits,_ = model(x)
                # [b]
                pred =  nn.functional.softmax(logits, dim=1)
                p = torch.tensor([pred[i, int(label[i].item())] for i in range(len(label))])

                # [b] vs [b] => scalar tensor
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = p
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, p), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TPFN = torch.eq(total_pred, total_label).float().sum(1)
            TPFN_prob = TPFN / cfg.NUM_TRANS

        # Eva for anormal data
        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for batchidx, (x, label) in pbar:
                x, label = x.to(device), label.to(device)

                # [b, 72]
                logits,_ = model(x)
                # [b]
                pred =  nn.functional.softmax(logits, dim=1)
                p = torch.tensor([pred[i, int(label[i].item())] for i in range(len(label))])

                # [b] vs [b] => scalar tensor
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = p
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, p), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TNFP = torch.eq(total_pred, total_label).float().sum(1)
            TNFP_prob = TNFP / cfg.NUM_TRANS

        TPFNTNFP_prob = torch.cat((TPFN_prob, TNFP_prob), 0)
        TPFNTNFP_label = torch.cat((torch.ones_like(TPFN), torch.zeros_like(TNFP)), 0)

        TPFNTNFP_label = torch.Tensor.cpu(TPFNTNFP_label)
        TPFNTNFP_prob = torch.Tensor.cpu(TPFNTNFP_prob)
    return TPFNTNFP_label, TPFNTNFP_prob


def cf_matrix(prob, label, threshold, name):
    pred = torch.where(prob >= threshold, torch.tensor(1), torch.tensor(0))
    print(pred)
    pred = torch.Tensor.cpu(pred)
    label = torch.Tensor.cpu(label)
    temp = torch.tensor(np.arange(0,len(pred)))
    print(temp[pred != label])
    conf_matrix = cf_matrix(prob, label, threshold)
    plt.ion()
    plt.imshow(np.array(conf_matrix), cmap=plt.cm.Blues)
    thresh = conf_matrix.max() / 2
    for x in range(2):
        for y in range(2):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black")

    # plt.tight_layout()
    plt.yticks(range(2), ['anormal', 'normal'])
    plt.xticks(range(2), ['anormal', 'normal'], rotation=45)
    plt.savefig("ConfusionMatrix"+name+'.png', bbox_inches='tight')
    plt.ioff()

    return confusion_matrix(label, pred)


def draw_roc(label, prob, name):
    label = label
    prob = prob
    fpr, tpr, threshold = roc_curve(label, prob)
    print(threshold)
    roc_auc = auc(fpr, tpr)
    plt.ion()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve AUC=" + str(roc_auc))
    plt.legend(loc="lower right")
    plt.savefig("ROCCurve"+name+'.png')
    plt.show()
    plt.close()
    plt.ioff()
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    best_threshold = threshold[maxindex]
    return best_threshold


name = 'Speed300'
label, prob = load_data('Speed300')
print(label, prob)
threshold = draw_roc(label, prob, 'Speed300')
print(threshold)
conf_matrix = cf_matrix(prob, label, threshold, 'Speed300')

# plt.imshow(np.array(conf_matrix), cmap=plt.cm.Blues)
# thresh = conf_matrix.max() / 2
# for x in range(2):
#     for y in range(2):
#         info = int(conf_matrix[y, x])
#         plt.text(x, y, info,
#                  verticalalignment='center',
#                  horizontalalignment='center',
#                  color="white" if info > thresh else "black")

# # plt.tight_layout()
# plt.yticks(range(2), ['anormal', 'normal'])
# plt.xticks(range(2), ['anormal', 'normal'], rotation=45)
# plt.savefig("ConfusionMatrix(AnormaliesSpeed300e1).png", bbox_inches='tight')
# plt.ioff()
# plt.show()
