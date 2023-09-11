# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/4/30 14:19
@author: LiFan Chen
@Filename: mol_featurizer.py
@Software: PyCharm
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import random
import torch
import os
from rdkit.Chem import AllChem
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import os
import torch
import pickle

num_atom_feat = 34


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    degree = [0, 1, 2, 3, 4, 5, 'other']
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding_unk(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    # return np.array(adjacency,dtype=np.float32)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def get_fingerprints(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    fp2str = fp.ToBitString()
    fp2array = np.array(list(fp2str), dtype=int)
    return fp2array


def fps2number(fpList):
    new_arr = np.zeros((len(fpList), 1024))
    for i, a in enumerate(fpList):
        new_arr[i, :] = np.array(list(a), dtype=int)
    return new_arr


def set_random_seed(seed=10):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def bulid_dataset(fileName,fileNameSuffix):
    with open(fileName, "r") as f:
        data_list = f.read().strip().split('\n')
    N = len(data_list)
    compounds, adjacencies, proteins = [], [], []
    model = Word2Vec.load("word2vec_50.model")
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequence = data.strip().split(" ")
        # print('smi_seq:',smiles,sequence)

        atom_feature, adj = mol_features(smiles)
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        protein = torch.FloatTensor(protein_embedding)

        compounds.append(atom_feature)
        adjacencies.append(adj)
        proteins.append(protein)
    dataset = list(zip(compounds, adjacencies, proteins))
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    with open("./dataset/"+fileNameSuffix, "wb") as f:
        pickle.dump(dataset, f)


def return_dataset(smi,seq):
    compounds, adjacencies, proteins = [], [], []
    model = Word2Vec.load("word2vec_50.model")
    smiles, sequence = smi.strip(),seq.strip()

    atom_feature, adj = mol_features(smiles)
    protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))

    atom_feature = torch.FloatTensor(atom_feature)
    adj = torch.FloatTensor(adj)
    protein = torch.FloatTensor(protein_embedding)

    compounds.append(atom_feature)
    adjacencies.append(adj)
    proteins.append(protein)
    dataset = list(zip(compounds, adjacencies, proteins))
    return dataset


if __name__ == "__main__":
    set_random_seed(1)
    with open('./data/test.txt', "r") as f:
        data_list = f.read().strip().split('\n')
    N = len(data_list)
    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []
    model = Word2Vec.load("word2vec_50.model")
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        smiles, sequence, interaction = data.strip().split(" ")

        atom_feature, adj = mol_features(smiles)
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
        label = np.array(interaction, dtype=np.float32)

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        protein = torch.FloatTensor(protein_embedding)
        label = torch.LongTensor(label)

        compounds.append(atom_feature)
        adjacencies.append(adj)
        proteins.append(protein)
        interactions.append(label)

    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    with open("dataset/test.txt", "wb") as f:
        pickle.dump(dataset, f)

