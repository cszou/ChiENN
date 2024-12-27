import torch
from torch_geometric.datasets import QM9
from tqdm import tqdm
from chienn.data.featurize import *

def preprocess_features(dataset):
    data_list = []
    idx = []
    for i, data in enumerate(tqdm(dataset)):
        smiles = data.smiles
        try:
            processed_data = smiles_to_data_with_circle_index(smiles)
            processed_data.y = data.y 
            data_list.append(processed_data)
        except:
            idx.append(i)
    return data_list


dataset = QM9(root='data/MoleculeNet/qm9')
mean = dataset.data.y.mean(dim=0)
std = dataset.data.y.std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std
processed_dataset = preprocess_features(dataset)
torch.save(processed_dataset, 'processed_qm9.pt')