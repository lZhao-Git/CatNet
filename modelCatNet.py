"""
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_recall_curve,auc,confusion_matrix
from Radam import *
from lookahead import Lookahead
from decimal import Decimal
import pandas as pd
from sklearn.metrics import roc_curve
from mol_featurizer import set_random_seed

set_random_seed(1)


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, self_attention, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        return trg


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, decoder_layer, self_attention, dropout, fc_dim,n_fc_layers,device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, self_attention, dropout, device) for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.fc_dim = fc_dim
        self.n_fc_layers = n_fc_layers
        self.fc_layers = nn.ModuleList()
        for i in range(self.n_fc_layers):
            if i == 0:
                self.fc_layers.append(self.fc_layer(dropout, hid_dim, fc_dim))
            else:
                self.fc_layers.append(self.fc_layer(dropout, fc_dim, fc_dim))
        self.fc_out = nn.Linear(fc_dim, 2)

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        for layer in self.layers:
            trg = layer(trg, src,trg_mask,src_mask)
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v

        for fc in self.fc_layers:
            sum = fc(sum)
        label = self.fc_out(sum)
        return label


class Predictor(nn.Module):
    def __init__(self, gat_dim,hid_dim,dropout,g_layers,alpha, encoder, decoder, device,atom_dim=34):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight_1 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.weight_2 = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()
        self.gat_dim = gat_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.alpha = alpha
        self.g_layers = g_layers
        self.gat_layers = [GATLayer(atom_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(g_layers)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * g_layers, gat_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(gat_dim, hid_dim)

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight_1.size(1))
        self.weight_1.data.uniform_(-stdv, stdv)
        self.weight_2.data.uniform_(-stdv, stdv)

    def comp_gat(self, atoms_vector, adj):
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    def forward(self, compound, adj,protein,atom_num,protein_num):
        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound = self.comp_gat(compound, adj)
        enc_src = self.encoder(protein)
        out = self.decoder(compound,enc_src, compound_mask, protein_mask)
        return out

    def __call__(self, data, train=True):
        Loss = nn.CrossEntropyLoss()
        if train:
            compound, adj, protein, correct_interaction, atom_num, protein_num = data
            predicted_interaction = self.forward(compound, adj,protein,atom_num,protein_num)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            compound, adj, protein, atom_num, protein_num = data
            predicted_interaction = self.forward(compound, adj,protein,atom_num,protein_num)
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_scores = ys[:, 1]
            return predicted_scores


def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    # fps_new = torch.zeros((N,1024), device=device)
    # i = 0
    # for fp in fps:
    #     fps_new[i,:] = fp
    #     i +=1
    # print('fp_new',fps_new.shape) # torch.Size([8, 1024])
    # return (atoms_new, adjs_new, fps_new,proteins_new, labels_new, atom_num, protein_num)
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)


def packData(atoms, adjs, proteins,device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms, proteins, labels = [], [], [], []
        data_num = 0
        for data in dataset:
            i = i+1
            atom, adj,protein, label = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels.append(label)
            if i % 8 == 0 or i == N:
                data_pack = pack(atoms, adjs, proteins, labels, device)
                loss = self.model(data_pack)
                loss.backward()
                adjs, atoms, proteins, labels = [], [], [], []
                loss_temp = loss.item() * data_pack[0].shape[0]
                data_num += data_pack[0].shape[0]
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss_temp
        return loss_total/data_num


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        loss_dev_total = 0
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,proteins, labels = [], [], [], []
                atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                loss_dev_cpu,correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                loss_dev_total += loss_dev_cpu
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return loss_dev_total/N,AUC, PRC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def testEvaluate(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,proteins, labels = [], [], [], []
                atom, adj,protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                loss_dev_cpu,correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)

        TN, FP, FN, TP = confusion_matrix(T, Y).ravel()
        roc_auc = roc_auc_score(T, S)
        accuracy = accuracy_score(T, Y)
        recall = recall_score(T, Y)
        specificity = TN / float(TN + FP)
        F1_score = f1_score(T, Y)
        ppv, se, _ = precision_recall_curve(T, S)
        PRC = auc(se, ppv)
        roc_auc, PRC,accuracy, recall, specificity, F1_score = map(lambda i: Decimal(i).quantize(Decimal('0.0001'), rounding='ROUND_HALF_UP'),
                                        [roc_auc, PRC,accuracy, recall, specificity, F1_score])
        return roc_auc, PRC,accuracy, recall, specificity, F1_score

    def testPredict(self, dataset):
        self.model.eval()
        N = len(dataset)
        S = []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,proteins = [], [], []
                atom, adj, protein = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                data = packData(atoms,adjs,proteins, self.model.device)
                predicted_scores = self.model(data, train=False)
                S.extend(predicted_scores)
        pred_prob_pd = pd.DataFrame(S, columns=['predictive probability'])
        S_label2 = [0 if i < 0.431 else 1 for i in S]
        pred_prob_pd_thred2 = pd.DataFrame(S_label2, columns=['predictive label'])
        pred_results = pd.concat([pred_prob_pd,pred_prob_pd_thred2], axis=1)
        return pred_results

    def find_best_cutoff(self, y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        best_index = np.argmax(tpr+(1-fpr)-1)
        best_threshold = thresholds[best_index]
        best_point = [fpr[best_index], tpr[best_index]]
        return best_threshold, best_point, fpr, tpr

    def valid_thd(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,proteins, labels = [], [], [], []
                atom, adj,protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                loss_dev_cpu,correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        best_threshold, best_point, fpr, tpr = self.find_best_cutoff(T, S)
        return best_threshold

    def testEvaluate_thd(self, dataset,best_threshold):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,proteins, labels = [], [], [], []
                atom, adj,protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                loss_dev_cpu,correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        Y = [0 if i < best_threshold else 1 for i in S]
        TN, FP, FN, TP = confusion_matrix(T, Y).ravel()
        roc_auc = roc_auc_score(T, S)
        accuracy = accuracy_score(T, Y)
        recall = recall_score(T, Y)
        specificity = TN / float(TN + FP)
        F1_score = f1_score(T, Y)
        ppv, se, _ = precision_recall_curve(T, S)
        PRC = auc(se, ppv)
        roc_auc, PRC,accuracy, recall, specificity, F1_score = map(lambda i: Decimal(i).quantize(Decimal('0.0001'), rounding='ROUND_HALF_UP'),
                                        [roc_auc, PRC,accuracy, recall, specificity, F1_score])
        return roc_auc, PRC,accuracy, recall, specificity, F1_score
    


