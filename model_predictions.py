"""
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
from modelCatNet import *
import timeit
import pickle
from mol_featurizer import set_random_seed


def test_prediction(varValue):
    set_random_seed(1)
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        # device = torch.device('cpu')
        # print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        # print('The code uses CPU!!!')

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

    """Start predicting"""
    print('Predicting...')
    best_model = os.listdir('bestModel')
    for name in best_model:
        if name.endswith('.pt'):
            k_v = name[:-3].split(',')
            # print(k_v)
            k_v_dict = {}
            for i in k_v:
                idx = i.find('=')
                k_v_dict[i[:idx]] = i[idx + 1:]
            lr = k_v_dict['lr']
            dropout = float(k_v_dict['dp'])
            weight_decay = k_v_dict['wd']
            kernel_size = int(k_v_dict['k'])
            n_layers =int(k_v_dict['n_l'])
            batch = int(k_v_dict['b'])
            hid_dim = int(k_v_dict['h_d'])
            n_heads = int(k_v_dict['h'])
            gat_dim = int(k_v_dict['g_d'])
            g_layers = int(k_v_dict['g_l'])
            step_size = int(k_v_dict['s'])
            alpha = float(k_v_dict['a'])
            fc_dim = int(k_v_dict['f_d'])
            n_fc_layers = int(k_v_dict['f_l'])

            encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
            decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, DecoderLayer, SelfAttention, dropout, fc_dim,
                              n_fc_layers, device)
            model = Predictor(gat_dim, hid_dim, dropout, g_layers, alpha, encoder, decoder, device)
            model.to(device)
            model.load_state_dict(torch.load(os.path.join('bestModel', name)))
            tester = Tester(model)

            """Load preprocessed data."""
            if isinstance(varValue, str):
                with open('./dataset/' + varValue, "rb") as f1:
                    dataset_test = pickle.load(f1)
                    test_pred = tester.testPredict(dataset_test)
                    print(test_pred)
                    test_pred.to_csv(varValue[:-4] + 'Pred.csv', index=False)
            elif isinstance(varValue, list):
                test_pred = tester.testPredict(varValue)
                print(test_pred)
            print('finish prediction')
        return test_pred


if __name__ == '__main__':
    test_prediction('demo1.txt')



