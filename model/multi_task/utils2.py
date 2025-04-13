#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
from itertools import combinations
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.utils import add_self_loops
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import random
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # `inputs`는 이미 `sigmoid`가 적용된 확률 값임.
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None and self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * loss
        else:
            F_loss = loss

        if self.reduction == 'none':
            return F_loss
        elif self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            raise ValueError(f"Invalid value for 'reduction': {self.reduction}")
            
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GraphNorm, GlobalAttention
from itertools import combinations
from torch_geometric.nn import MLP

class MultiTaskGAT(nn.Module):
    def __init__(self, num_features, n_heads1, output_dim_idx, dropout, num_tasks=4):
        super(MultiTaskGAT, self).__init__()

        self.conv2 = GATConv(num_features, num_features, heads=n_heads1, dropout=dropout)
        self.relu = nn.ReLU()
        self.graph_norm2 = GraphNorm(num_features * n_heads1)
        
        self.task_combinations_2 = list(combinations(range(num_tasks), 3))
        
        self.hidden_layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features * n_heads1, output_dim_idx),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in self.task_combinations_2
        ])
        
        self.final_layer = nn.ModuleList([nn.Linear(output_dim_idx, 1) for _ in range(num_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.attention_net = nn.Sequential(
            nn.Linear(num_features * n_heads1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.global_attention = GlobalAttention(gate_nn=self.attention_net)
        
        self.node_dropout = nn.Dropout(dropout)
        
        self.weight_matrix = nn.Parameter(torch.ones(len(self.task_combinations_2)))

    def forward(self, data, selected_tasks):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x2 = self.conv2(x, edge_index)
        x = self.graph_norm2(x2, batch)
        x = self.relu(x)
        x = self.node_dropout(x)

        x = self.global_attention(x, batch)

        outputs = {task: [] for task in selected_tasks}
        for idx, (layer, combo) in enumerate(zip(self.hidden_layer2, self.task_combinations_2)):
            if set(combo).issubset(selected_tasks):
                output = layer(x)
                weight = self.weight_matrix[idx]
                for task in combo:
                    outputs[task].append(output * weight)
        
        final_outputs = []
        for task in selected_tasks:
            if outputs[task]:
                combined_output = torch.sum(torch.stack(outputs[task]), dim=0) / len(outputs[task])
                final_output = self.final_layer[task](combined_output).view(-1)
                final_outputs.append(self.sigmoid(final_output))
            else:
                raise IndexError(f"Task {task} has no corresponding output in outputs dictionary.")

        return x, tuple(final_outputs), (edge_index, None)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, val_loss)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, optimizer, val_loss):
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss_min': val_loss,
        }
        torch.save(state, self.path)
        self.val_loss_min = val_loss

class WeightTracker:
    def __init__(self):
        self.initial_weights = {}

    def save_initial_weights(self, model):
        model_state_dict = model.state_dict()
        for name, param in model_state_dict.items():
            self.initial_weights[name] = param.clone()

    def compare_weights_after_training(self, model):
        model_state_dict = model.state_dict()
        for name, param in model_state_dict.items():
            if name in self.initial_weights:
                if torch.equal(param, self.initial_weights[name]):
                    print(f"Layer {name} is frozen (weights unchanged after training).")
                else:
                    print(f"Layer {name} has changed (weights have been updated during training).")

def split_dataset_by_combinations(dataset, task_count):
    from torch_geometric.data import DataLoader, Batch
    combination_dataloaders = {}
    two_task_combos = list(combinations(range(task_count), 2))
    for combo in two_task_combos:
        indices_2_task = []
        for idx in range(len(dataset)):
            data_item = dataset[idx]
            y = data_item.y.squeeze().numpy()
            if all(y[j] != -1 for j in combo):
                indices_2_task.append(idx)

        if len(indices_2_task) > 0:
            filtered_dataset = Subset(dataset, indices_2_task)
            combination_dataloaders[combo] = DataLoader(
                filtered_dataset, batch_size=16, shuffle=True, collate_fn=Batch.from_data_list, 
                num_workers=0)
            print(f"Combination {combo}: {len(indices_2_task)} samples")
        else:
            print(f"Combination {combo}: No samples found.")

    return combination_dataloaders

def save_checkpoint(model, optimizer, filename, layers_to_save=None):
    state_dict = model.state_dict()
    if layers_to_save:
        filtered_state_dict = {k: v for k, v in state_dict.items() if any(layer in k for layer in layers_to_save)}
    else:
        filtered_state_dict = state_dict
    
    state = {
        'state_dict': filtered_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename} with keys: {state.keys()}")

def load_all_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(f"\nLoading all weights from '{checkpoint_path}'...")
    if 'state_dict' not in checkpoint:
        raise KeyError(f"Expected 'state_dict' key in the checkpoint, but not found.")
    
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    print("All layers loaded successfully.")

def compare_full_model_weights(checkpoint_paths, model):
    model_state_dict = model.state_dict()
    for layer_num, checkpoint_path in enumerate(checkpoint_paths, start=1):
        checkpoint = torch.load(checkpoint_path)
        checkpoint_state_dict = checkpoint['state_dict']
        for name, param in checkpoint_state_dict.items():
            if name in model_state_dict:
                if not torch.equal(param, model_state_dict[name]):
                    print(f"Mismatch in weights for {name} in layer {layer_num}")
            else:
                print(f"{name} not found in the model's state_dict")

from collections import defaultdict

from collections import defaultdict
import torch

def train(epoch, model, criterion, optimizer, device, train_loader, selected_tasks, idx_to_task, time_count=False):
    import time
    if time_count:
        start = time.time()

    model.train()
    task_losses = defaultdict(float)
    num_batches = defaultdict(int)

    for batch in train_loader:
        data = batch.to(device)

        list_labels = [batch.y[:, i].to(device) for i in range(len(selected_tasks))]
        _, list_task, _ = model(data, selected_tasks)

        optimizer.zero_grad()
        batch_loss = 0.0

        for i, task in enumerate(selected_tasks):
            task_pred = list_task[i]
            mask = list_labels[i] != -1  
            if mask.sum() == 0:
                continue
            total_loss = criterion(task_pred[mask], list_labels[i][mask])

            task_losses[f"task_{task}"] += total_loss.item()
            num_batches[f"task_{task}"] += 1
            batch_loss += total_loss

        batch_loss.backward()
        optimizer.step()

    avg_task_losses = {}
    for task in selected_tasks:
        if num_batches[f"task_{task}"] > 0:
            avg_loss = task_losses[f"task_{task}"] / num_batches[f"task_{task}"]
            avg_task_losses[task] = avg_loss

    if avg_task_losses:
        train_loss = sum(avg_task_losses.values()) / len(avg_task_losses)
    else:
        train_loss = 0.0

    if time_count:
        end = time.time()
        print(f"Epoch {epoch} training time: {end - start:.2f} seconds")

    return train_loss

def validation(epoch, model, criterion, device, validation_loader, selected_tasks, idx_to_task):
    model.eval()
    task_losses = defaultdict(float)
    num_batches = defaultdict(int)

    with torch.no_grad():
        for batch in validation_loader:
            data = batch.to(device)
            list_labels = [batch.y[:, i].to(device) for i in range(len(selected_tasks))]
            _, list_task, _ = model(data, selected_tasks)

            for i, task in enumerate(selected_tasks):
                task_pred = list_task[i]
                mask = list_labels[i] != -1 
                if mask.sum() == 0:
                    continue
                total_loss = criterion(task_pred[mask], list_labels[i][mask])
                task_losses[f"task_{task}"] += total_loss.item()
                num_batches[f"task_{task}"] += 1

    avg_task_losses = {}
    for task in selected_tasks:
        if num_batches[f"task_{task}"] > 0:
            avg_loss = task_losses[f"task_{task}"] / num_batches[f"task_{task}"]
            avg_task_losses[task] = avg_loss

    if avg_task_losses:
        val_loss = sum(avg_task_losses.values()) / len(avg_task_losses)
    else:
        val_loss = 0.0

    return val_loss

class CustomDataset(TorchDataset):
    def __init__(self, data_x, data_y, smiles_list):
        self.data_x = data_x
        self.data_y = data_y
        self.smiles_list = smiles_list

    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        c_size = torch.tensor(self.data_x[idx][0], dtype=torch.int64)
        features = torch.tensor(self.data_x[idx][1], dtype=torch.float32)
        edge_index = torch.tensor(self.data_x[idx][2], dtype=torch.int64).t()
        edge_count = torch.tensor(len(self.data_x[idx][2]), dtype=torch.int64)  # 엣지 수 추가
        label = torch.tensor(self.data_y[idx], dtype=torch.float32).unsqueeze(0)
        smiles = self.smiles_list[idx]
        return Data(x=features, edge_index=edge_index, edge_count=edge_count, y=label, smiles=smiles, c_size=c_size)
    
def count_samples_for_dataset(data_y_np):
    counts = []

    for column in data_y_np.T:  
        label_counts = {
            -1: np.sum(column == -1),
            0: np.sum(column == 0),
            1: np.sum(column == 1),
            'total': len(column)
        }
        counts.append(label_counts)

    return counts

def load_data(batch_size=8):
    data_x_np = np.load('/../../data/multi task/val/multi_refined_data.npy', allow_pickle=True)
    data_y_np = np.load('/../../data/multi task/val/multi_refined_labels.npy', allow_pickle=True)

    path = '/../../data/multi task/val/val_smiles.txt'

    with open(path, 'r', encoding='utf-8') as f:
        val_smiles = [line.rstrip() for line in f] 
        
    data_y_np = data_y_np.T

    dataset = CustomDataset(data_x_np, data_y_np, val_smiles)    

    return dataset

