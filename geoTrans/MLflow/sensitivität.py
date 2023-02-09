import mlflow
import torch
from get_dataloader import get_train_test_loader
from Multi_Model_noGeo import WideResNet
from matplotlib import pyplot as plt
from train_test import test_model
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from metric import draw_roc, cf_matrix, get_f1, draw_pr

batchsz = 128

mlflow.set_tracking_uri('http://localhost:5001')
path = "models:/Multimodel_Best/1"
model = mlflow.pytorch.load_model(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader50 = get_train_test_loader(batchsz, 64, 'Train50', 'Vali50')
_, test_loader100 = get_train_test_loader(batchsz, 64, 'Train100', 'Vali100')
_, test_loader200 = get_train_test_loader(batchsz, 64, 'Train200', 'Vali200')
_, test_loader300 = get_train_test_loader(batchsz, 64, 'Train300', 'Vali300')
_, test_loader30 = get_train_test_loader(batchsz, 64, 'Train30', 'Vali30')

anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader50)
fpr50, tpr50, threshold = roc_curve(total_label, total_pred)
anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader100)
fpr100, tpr100, threshold = roc_curve(total_label, total_pred)
anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader200)
fpr200, tpr200, threshold = roc_curve(total_label, total_pred)
anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader300)
fpr300, tpr300, threshold = roc_curve(total_label, total_pred)
anormalacc, total_label, total_pred, avg_val_loss = test_model(model, device, test_loader30)
fpr30, tpr30, threshold = roc_curve(total_label, total_pred)
best_threshold = draw_roc(total_label, total_pred, "roc30")
cf_matrix(total_pred, total_label, best_threshold, "cf30")
auc_score = auc(fpr30, tpr30)
print(auc_score)
fig, ax = plt.subplots()

ax.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Test for Sensitivity")
ax5 = ax.plot(fpr30, tpr30, label='ROC curve 0')
# plt.legend(loc="lower right")
ax1 = ax.plot(fpr50, tpr50, label='ROC curve 1')
ax2 = ax.plot(fpr100, tpr100, label='ROC curve 2')
ax3 = ax.plot(fpr200, tpr200, label='ROC curve 3')
ax4 = ax.plot(fpr300, tpr300, label='ROC curve 4')

ax.legend()
# plt.legend([ax1, ax2, ax3, ax4],['ROC curve 1', 'ROC curve 2', 'ROC curve 3', 'ROC curve 4'])
plt.savefig('sensitivität'+'.png')
