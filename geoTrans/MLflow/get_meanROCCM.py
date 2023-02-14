import mlflow
import torch
from get_dataloader import get_train_test_loader
from Multi_Model_noGeo import WideResNet
from matplotlib import pyplot as plt
from train_test import test_model
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import numpy as np
from metric import draw_roc, cf_matrix, get_f1, draw_pr

batchsz = 128
fpr = []
tpr = []
cf = []

mlflow.set_tracking_uri('http://localhost:5001')
path = "models:/Multimodel_Best/1"
model1 = mlflow.pytorch.load_model(path)
path = "models:/Multimodel_Best/2"
model2 = mlflow.pytorch.load_model(path)
path = "models:/Multimodel_Best/3"
model3 = mlflow.pytorch.load_model(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader50 = get_train_test_loader(batchsz, 64, 'Train50', 'Vali50')
mean_fpr = np.linspace(0, 1, 100)

anormalacc, total_label, total_pred, avg_val_loss = test_model(model1, device, test_loader50)
fpr501, tpr501, threshold = roc_curve(total_label, total_pred)
interp_tpr = np.interp(mean_fpr, fpr501, tpr501)
interp_tpr[0] = 0.0
tpr.append(interp_tpr)

best_threshold = draw_roc(total_label, total_pred, "roc501")
_, cf1 = cf_matrix(total_pred, total_label, best_threshold, "cf501")
cf.append(cf1)

anormalacc, total_label, total_pred, avg_val_loss = test_model(model2, device, test_loader50)
fpr502, tpr502, threshold = roc_curve(total_label, total_pred)
interp_tpr = np.interp(mean_fpr, fpr502, tpr502)
interp_tpr[0] = 0.0
tpr.append(interp_tpr)

best_threshold = draw_roc(total_label, total_pred, "roc502")
_, cf2 = cf_matrix(total_pred, total_label, best_threshold, "cf502")
cf.append(cf2)

anormalacc, total_label, total_pred, avg_val_loss = test_model(model3, device, test_loader50)
fpr503, tpr503, threshold = roc_curve(total_label, total_pred)
interp_tpr = np.interp(mean_fpr, fpr503, tpr503)
interp_tpr[0] = 0.0
tpr.append(interp_tpr)

best_threshold = draw_roc(total_label, total_pred, "roc503")
_, cf3 = cf_matrix(total_pred, total_label, best_threshold, "cf503")
cf.append(cf3)

fig, ax = plt.subplots()
mean_tpr = np.mean(tpr, axis=0)
mean_tpr[-1] = 1.0

ax.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")

# plt.legend(loc="lower right")
ax1 = ax.plot(fpr501, tpr501, label='ROC curve 1', color="g", alpha=0.3, lw=1)
ax2 = ax.plot(fpr502, tpr502, label='ROC curve 2', color="g", alpha=0.3, lw=1)
ax3 = ax.plot(fpr503, tpr503, label='ROC curve 3', color="g", alpha=0.3, lw=1)
ax4 = ax.plot(mean_fpr, mean_tpr, label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (0.916, 0.016),lw=2)

std_tpr = np.std(tpr, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)


ax.legend(fontsize=15)
plt.savefig('roc3'+'.png')
plt.close()


mean_cf = np.mean(cf, axis=0)
std_cf = np.std(cf, axis=0)
plt.imshow(mean_cf, cmap=plt.cm.Blues)
thresh = mean_cf.max() / 2
for x in range(2):
    for y in range(2):
        info = r'%0.2f $\pm$ %0.2f' % (mean_cf[y,x],std_cf[y,x])
        plt.text(x, y, info,
                verticalalignment='center',
                horizontalalignment='center',
                color="white" if mean_cf[y,x] > thresh else "black", fontsize=15)

# plt.tight_layout()
plt.yticks(range(2), ['anormal', 'normal'], fontsize=15)
plt.xticks(range(2), ['anormal', 'normal'], rotation=0, fontsize=15)
plt.savefig('mean'+'.png', bbox_inches='tight')
mlflow.log_artifact('mean'+'.png')
plt.ioff()
plt.clf()
