import torch

from torch import nn
from tqdm import tqdm
import mlflow

def test_model(model, device, test_loader):
    parameter_num = 2
    model.eval()
    total_correct = 0
    total_num = 0
    num_batches = len(test_loader)
    criterion = nn.BCEWithLogitsLoss().to(device)
    val_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batchidx, (x1, x2, label) in pbar:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            #  [b, 1]
            logits = model(x1, x2)
            val_loss += criterion(logits, label).item()
            prob = torch.sigmoid(logits)
            # prediction
            pred = torch.where(prob>0.5, torch.ones_like(prob).to(device), torch.zeros_like(prob).to(device))
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x1.size(0)
            if batchidx == 0:
                total_pred = prob
                total_label = label
            else:
                total_pred = torch.cat((total_pred, prob), 0)
                total_label = torch.cat((total_label, label), 0)

        total_label = torch.Tensor.cpu(total_label)
        total_pred = torch.Tensor.cpu(total_pred)
        anormalacc = total_correct / total_num
        avg_val_loss = val_loss / num_batches

    return anormalacc, torch.ones_like(total_label) - total_label, torch.ones_like(total_pred) - total_pred, avg_val_loss

#  train only one loop
def train_model(model, device, train_loader, optimizer, epoch, epochs, global_step):
    model.train()
    parameter_num = 2
    train_loss = 0.0
    num_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    criterion = nn.BCEWithLogitsLoss().to(device)
    for batchidx,  (x1, x2, label) in pbar:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        logits = model(x1, x2)
        loss = criterion(logits, label)
        mlflow.log_metric("train_loss", loss.item(), global_step)
        global_step += 1
        pred = torch.where(logits>0.0, torch.ones_like(logits).to(device), torch.zeros_like(logits).to(device))
        correct = torch.eq(pred, label).float().sum().item()
        train_loss += loss.item()
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Epoch [{epoch}/{int(epochs)}]')
        pbar.set_postfix({'loss=': loss.item(), 'acc=': correct / x1.size(0)})
    avg_train_loss = train_loss / num_batches
    return global_step, avg_train_loss

# from get_dataloader import get_train_test_loader, MyTrainDataLoaderIter, MyValDataLoaderIter
# from Multi_Model_noGeo import WideResNet, ModelWrapper
# import torch.optim as optim
# from sklearn.metrics import roc_curve, auc
# from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
# from matplotlib import pyplot as plt

# lr = 0.00001
# batchsz = 128
# epochs = 15
# device = torch.device('cuda')
# train_loader, test_loader = get_train_test_loader(batchsz, 64, 'Train50', 'Vali50')
#  # Prepare wrapper for dataloader
# trainloader_wrapper = MyTrainDataLoaderIter(train_loader)
# valloader_wrapper = MyValDataLoaderIter(test_loader)
#   # Prepare wrapper for model
# model = WideResNet(10, 1, 6, 5, 0, 0, 1)
# model_wrapper = ModelWrapper(model)
# print(model)
# model_wrapper = model_wrapper.to(device)
# torch.manual_seed(1234)
# x1, x2, label = iter(test_loader).next()
# print('x1:', x1.shape, 'x2:', x2.shape, 'label:', label.shape)
# x1, x2, label = iter(test_loader).next()
# print('x1:', x1.shape, 'x2:', x2, 'label:', label)

# mlflow.set_tracking_uri('http://localhost:5001')
# path = "models:/Multimodel_test0/1"
# model = mlflow.pytorch.load_model(path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# anormalacc,_,_,_ = test_model(model, device, test_loader)
# print(anormalacc)

# # torch.manual_seed(1234)
# optimizer = optim.Adam(model_wrapper.parameters(), lr=0.00001, weight_decay=5e-4)
# criterion = nn.BCEWithLogitsLoss()
# lr_finder = LRFinder(model_wrapper, optimizer, criterion, device="cuda")
# lr_finder.range_test(trainloader_wrapper, end_lr=100, num_iter=100)
# fig, ax = plt.subplots()
# _, bestlr=lr_finder.plot(ax=ax)
# # ax can be further manipulated by the user after lr_finder.plot
# fig.savefig("test.png")
# lr_finder.reset()
# # global_step = 0
# best_auc = 0

# optimizer = optim.Adam(model.to(device).parameters(), lr=bestlr, weight_decay=5e-4)
# for epoch in range(epochs):
#     print(optimizer.state_dict()['param_groups'][0]['lr'])
#     avg_train_loss = train(model, device, train_loader, optimizer, epoch, epochs)
#     anormalacc, total_label, total_pred = test(model, device, test_loader)
#     fpr, tpr, threshold = roc_curve(total_label, total_pred)
#     auc_score = auc(fpr, tpr)
#     if auc_score > best_auc:
#         best_auc = auc_score
#         print(epoch, best_auc, threshold)

#     print(epoch, 'anormal acc:', anormalacc)
