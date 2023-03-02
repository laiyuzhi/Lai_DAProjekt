#from dataset_bioreaktor import Bioreaktor_Detection
import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')
from dataset_bioreaktorSpeed import Bioreaktor_Detection
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
from visdom import Visdom

def main():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData2'

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    torch.manual_seed(1234)
    batchsz = cfg.BATCH_SIZE
    lr = cfg.LEARN_RATE
    epochs = cfg.EPOCHS
    num_trans = cfg.NUM_TRANS

    train_db = Bioreaktor_Detection(root, 64, mode='train')
    testbig_db = Bioreaktor_Detection(root, 64, mode='testbig')
    testsmall_db = Bioreaktor_Detection(root, 64, mode='testsmall')
    vali_db = Bioreaktor_Detection(root, 64, mode='vali')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    testbig_loader = DataLoader(testbig_db, batch_size=batchsz, num_workers=0)
    testsmall_loader = DataLoader(testsmall_db, batch_size=batchsz, num_workers=0)
    vali_loader = DataLoader(vali_db, batch_size=batchsz, num_workers=0)
    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    model = WideResNet(16, num_trans, 8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    print(model)
    # Visualisation through visdom
    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0, 0.0, 0.0]], [0.], win='test_acc', opts=dict(title='normal acc.&anormal acc.',
                                                   legend=['normal acc.', 'anormal_big acc.', 'anormal_small acc.']))

    
    global_step = 0
    for epoch in range(int(np.ceil(epochs/num_trans))):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batchidx, (x, label) in pbar:
            # pbar.set_description("Epoch: s%" % str(epoch))
            # [b, 3, 64, 64]
            x = x.to(device)
            label = label.to(device)

            logits,_ = model(x)
            loss = criterion(logits, label)

            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            viz.line([loss.item()], [global_step], win='train_loss', update='append')

            pbar.set_description(f'Epoch [{epoch}/{int(np.ceil(epochs/num_trans))}]')
            pbar.set_postfix({'loss=': loss.item(), 'acc=': correct/x.size(0)})

        print(epoch, 'loss:', loss.item())
        path = 'modelspeed2168' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), path)

        if epoch != 0 and epoch % cfg.VAL_EACH == 0 or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
                for batchidx, (x, label) in pbar:
                    x, label = x.to(device), label.to(device)

                    # [b, 72]
                    logits,_ = model(x)
                    # [b]
                    pred = logits.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc_norm = total_correct / total_num
                print(epoch, 'anormalnorm acc:', acc_norm)
        ##test
        if epoch != 0 and epoch % cfg.VAL_EACH == 0 or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testbig_loader), total=len(testbig_loader))
                for batchidx, (x, label) in pbar:
                    x, label = x.to(device), label.to(device)

                    # [b, 72]
                    logits,_ = model(x)
                    # [b]
                    pred = logits.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc_big = total_correct / total_num
                print(epoch, 'anormalbig acc:', acc_big)
        # scheduler.step()
        if epoch != 0 and epoch % cfg.VAL_EACH == 0 or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testsmall_loader), total=len(testsmall_loader))
                for batchidx, (x, label) in pbar:
                    x, label = x.to(device), label.to(device)

                    # [b, 72]
                    logits,_ = model(x)
                    # [b]
                    pred = logits.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc_small = total_correct / total_num
                print(epoch, 'anormalsmall acc:', acc_small)
        # scheduler.step()
        viz.line([[acc_norm, acc_big, acc_small]], [global_step], win='test_acc', update='append')


if __name__ == '__main__':
    main()
