import torch

from dataset_bioreaktorMulti_NoGeo import Bioreaktor_Detection as dataset
from  torch.utils.data import DataLoader
from torch_lr_finder import TrainDataLoaderIter, ValDataLoaderIter

def get_train_test_loader(batchsz=128, resize=64, csv_train='Train50', csv_test='Vali50'):
    root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll"
    train_db = dataset(root, resize, mode='Train', train_name=csv_train)
    test_db = dataset(root, resize, mode='Vali', test_name=csv_test)

    train_loader = DataLoader(train_db, batchsz, shuffle=True)
    test_loader = DataLoader(test_db, batchsz, shuffle=False)


    return train_loader, test_loader


class MyTrainDataLoaderIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        # Since the data we want to used are the first two values of
        # `batch_data`, we need to pack them in this format,
        *desired_data, label = batch_data   # desired_data: x, x_embds
        return desired_data, label


class MyValDataLoaderIter(ValDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        *desired_data, label = batch_data
        return desired_data, label
