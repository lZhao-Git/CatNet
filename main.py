"""
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import pickle
from mol_featurizer import set_random_seed


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    set_random_seed(1)
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    with open('./dataset/train.txt',"rb") as f1:
        data_train = pickle.load(f1)
    dataset_train = shuffle_dataset(data_train, 1234)
    with open('./dataset/validation.txt',"rb") as f2:
        data_validation = pickle.load(f2)
    dataset_dev = shuffle_dataset(data_validation, 1234)
    print('dataset:',len(dataset_train),len(dataset_dev)) # 1450 1160 290

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 256
    n_layers = 2
    n_heads = 8
    dropout = 0.1
    batch = 32
    lr = 5e-4
    weight_decay = 1e-5
    iteration = 300
    kernel_size = 1
    gat_dim = 64
    g_layers = 3
    alpha = 0.05
    fc_dim = 256
    n_fc_layers = 2
    step_size = 10

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, DecoderLayer, SelfAttention, dropout,fc_dim,n_fc_layers, device)
    model = Predictor(gat_dim,hid_dim,dropout,g_layers,alpha,encoder, decoder, device)
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'result/lr=5e-4,dp=0.1,wd=1e-5,k=1,n_l=2,b=32,h_d=256,h=8,g_d=64,g_l=3,s=10,a=0.05,f_d=256,f_l=2.txt'
    file_model = 'model/lr=5e-4,dp=0.1,wd=1e-5,k=1,n_l=2,b=32,h_d=256,h=8,g_d=64,g_l=3,s=10,a=0.05,f_d=256,f_l=2.pt'
    AUC = ('Epoch\tTime\tLoss_train\tLoss_dev\tAUC_dev\tPRC_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """Start training."""
    print('Training...')
    print(AUC)
    start = timeit.default_timer()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
    num = 0
    max_AUC_dev = 0
    for epoch in range(1, iteration + 1):
        loss_train = trainer.train(dataset_train, device)
        loss_dev,AUC_dev, PRC_dev = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,loss_dev, AUC_dev,PRC_dev]
        print('\t'.join(map(str, AUCs)))
        scheduler.step()
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
            num = 0
        else:
            num += 1
            print('the performance metric not improved in {} epochs on the valid set'.format(num))
            if num >= 10:
                break

