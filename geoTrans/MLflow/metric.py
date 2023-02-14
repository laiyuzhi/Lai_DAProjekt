import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import mlflow
import numpy as np


def get_f1(precision, recall):
    f1_scores = (2 * precision * recall) / (precision + recall+1e-5)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    return best_f1_score


def draw_pr(label, prob, name):
    prob, label = torch.Tensor.cpu(prob), torch.Tensor.cpu(label)
    precision, recall, thresholds_pr = precision_recall_curve(label, prob)
    plt.ion()
    plt.plot(precision, recall, label='PR curve')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title("PR Curve for Multimodel")
    plt.legend(loc="lower right")
    plt.savefig(name+'.png')
    mlflow.log_artifact(name+'.png')
    plt.ioff()
    plt.clf()



def cf_matrix(prob, label, threshold, name):
    prob, label = torch.Tensor.cpu(prob), torch.Tensor.cpu(label)
    pred = torch.where(prob >= threshold, torch.tensor(1), torch.tensor(0))
    pred = torch.Tensor.cpu(pred)
    label = torch.Tensor.cpu(label)
    temp = torch.tensor(np.arange(0,len(pred)))
    conf_matrix = confusion_matrix(label, pred)
    plt.imshow(np.array(conf_matrix), cmap=plt.cm.Blues)
    thresh = conf_matrix.max() / 2
    for x in range(2):
        for y in range(2):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color="white" if info > thresh else "black", fontsize=24)

    # plt.tight_layout()
    plt.yticks(range(2), ['anormal', 'normal'], fontsize=15)
    plt.xticks(range(2), ['anormal', 'normal'], rotation=0, fontsize=15)
    plt.savefig(name+'.png', bbox_inches='tight')
    mlflow.log_artifact(name+'.png')
    plt.ioff()
    plt.clf()
    return temp[pred.view(-1) != label.view(-1)], conf_matrix


def draw_roc(label, prob, name):
    prob, label = torch.Tensor.cpu(prob), torch.Tensor.cpu(label)
    fpr, tpr, threshold = roc_curve(label, prob)
    plt.ion()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title("ROC Curve for Multimodel")
    plt.legend(loc="lower right")
    plt.savefig(name+'.png')
    mlflow.log_artifact(name+'.png')
    plt.ioff()
    plt.clf()
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    best_threshold = threshold[maxindex]
    return best_threshold
