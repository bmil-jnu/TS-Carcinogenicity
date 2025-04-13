#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_add_pool as gap
from itertools import combinations
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention, GraphNorm

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
            
class single_task(nn.Module):
    def __init__(self, num_features, n_heads1, output_dim, dropout):
        super(single_task, self).__init__()

        self.conv2 = GATConv(num_features, num_features, heads=n_heads1, dropout=dropout)
        self.relu = nn.ReLU()
        self.graph_norm2 = GraphNorm(num_features * n_heads1)
        
        self.fc_g1 = torch.nn.Linear(num_features * n_heads1, output_dim)
        self.dropout = nn.Dropout(p=dropout)
#         self.fc_g2 = torch.nn.Linear(output_dim , hidden_size)
        self.out = nn.Linear(output_dim, 1)  
        self.sigmoid = nn.Sigmoid()
        
            # 어텐션 풀링
        self.attention_net = nn.Sequential(
            nn.Linear(num_features * n_heads1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.global_attention = GlobalAttention(gate_nn=self.attention_net)
        self.node_dropout = nn.Dropout(dropout)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GATConv 레이어 처리
        x2, w = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.graph_norm2(x2, batch)
        x = self.relu(x)
        x = self.node_dropout(x)

        x1 = self.global_attention(x, batch)
        
        x = self.fc_g1(x1)
        x = self.relu(x)
        x = self.dropout(x) 
        
        out = self.out(x)
        
        # Apply sigmoid activation
        out = torch.sigmoid(out)
        
        return out, w

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 학습 함수 정의
def train_single_task(epoch, model, criterion, optimizer, device, train_loader, time_count=False):
    if time_count:
        start = time.time()
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        data = batch.to(device)
        labels = batch.y.to(device)
        optimizer.zero_grad()
        outputs, _ = model(data)
        outputs = outputs.view_as(labels)
        
        mask = labels != -1
        if mask.sum() == 0:  # 모든 데이터가 -1 값인 경우
            continue

        loss = criterion(outputs[mask], labels[mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
#     print(f'====> Epoch: {epoch} Average Train Loss: {avg_loss:.4f}')
    
    if time_count:
        end = time.time()
        print(f"Epoch {epoch} training took {end - start:.2f} seconds")
    
    return avg_loss

# 검증 함수 정의
def val_single_task(epoch, model, criterion, device, validation_loader, time_count=False):
    if time_count:
        start = time.time()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            data = batch.to(device)
            labels = batch.y.to(device)
            outputs, _ = model(data)
            outputs = outputs.view_as(labels)
            
            mask = labels != -1
            if mask.sum() == 0:
                continue
            
            loss = criterion(outputs[mask], labels[mask])
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
#     print(f'====> Epoch: {epoch} Average Validation Loss: {avg_loss:.4f}')
    
    if time_count:
        end = time.time()
        print(f"Epoch {epoch} validation took {end - start:.2f} seconds")
    
    return avg_loss

# 검증 함수 정의
def test_single_task(epoch, model, criterion, device, test_loader, time_count=False):
    if time_count:
        start = time.time()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch.to(device)
            labels = batch.y.to(device)
            outputs, _ = model(data)
            outputs = outputs.view_as(labels)
            
            mask = labels != -1
            if mask.sum() == 0:
                continue
            
            loss = criterion(outputs[mask], labels[mask])
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f'====> Epoch: {epoch} Average test Loss: {avg_loss:.4f}')
    
    if time_count:
        end = time.time()
        print(f"Epoch {epoch} test took {end - start:.2f} seconds")
    
    return avg_loss

    
def count_samples_per_label(labels):
    total_count = len(labels)
    without_minus_one_count = (labels != -1).sum().item()
    return total_count, without_minus_one_count

def print_sample_counts(total, without_minus_one, phase, epoch, always_print=False):
    if always_print or epoch == 1:
        print(f"\nSample counts during {phase}:")
        print(f"Total = {total}, Without -1 = {without_minus_one}")
        print()

class CustomDataset(TorchDataset):  # PyTorch의 기본 Dataset 클래스를 상속
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        c_size = torch.tensor(self.data_x[idx][0], dtype=torch.int64)
        feature = torch.tensor(self.data_x[idx][1], dtype=torch.float32)
        edge_index = torch.tensor(self.data_x[idx][2], dtype=torch.int64).transpose(1, 0)
        label = torch.tensor(self.data_y[idx], dtype=torch.float32).unsqueeze(0)
        data = Data(x=feature, edge_index=edge_index, y=label)
        data.c_size = c_size
        return data

def load_data(task_name):
    # Paths
    base_path = "/data/home/dbswn0814/2025JCM/data/single task/{}/{}_{}.npy"
    
    # Load data for the given task_name

    data_x_np = np.load(base_path.format("val", "data", task_name), allow_pickle=True)
    data_y_np = np.load(base_path.format("val", "labels", task_name), allow_pickle=True)
    
    # Transpose labels
    data_y_np = data_y_np.T
  
    dataset = CustomDataset(data_x_np, data_y_np)
    
    return dataset