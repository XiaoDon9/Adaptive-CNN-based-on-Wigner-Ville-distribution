# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn.functional as F


def evaluate_cla(preds, labels):

    # Convert predictions to class indices if they are logits
    if preds.dim() > 1:
        preds = preds.argmax(dim=1)
    
    # Calculate accuracy using PyTorch
    acc = (preds == labels).float().mean().item()
    
    # Convert tensors to numpy arrays for f1_score calculation
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1 = f1_score(labels_np, preds_np, average='weighted')

    return acc, f1

def evaluate_reg(preds, labels):
    
    # Calculate MSE using PyTorch
    mse = F.mse_loss(preds, labels).item()
    
    # Calculate R2 score using PyTorch
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    ss_res = ((labels - preds) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    return mse, r2

def gpu(tensor, gpu=False):
    if gpu:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return tensor.to('mps')
        else:
            return tensor.cuda()
    else:
        return tensor

