# train.py
# Contains the training and evaluation loops.

import torch
import torch.nn as nn
# from tqdm import tqdm

import config

def train_one_epoch(model, loader, optimizer, loss_fn_class, loss_fn_corr, edge_index, device):
    """
    Runs one full epoch of training.
    """
    model.train()
    total_loss = 0.0
    total_class_loss = 0.0
    total_corr_loss = 0.0
    num_batches = 0

    for (input_pose, target_pose, label) in loader:
        input_pose = input_pose.to(device)
        target_pose = target_pose.to(device)
        label = label.to(device)

        # Create the batch vector required for PyG global_mean_pool
        B = input_pose.shape[0] # Batch size
        N = input_pose.shape[1] # Number of nodes (57)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)

        optimizer.zero_grad()
        pred_logits, pred_corrected_pose = model(input_pose, edge_index, batch_vec, labels=label, use_feedback=True)

        loss_class = loss_fn_class(pred_logits, label)
        loss_corr = loss_fn_corr(pred_corrected_pose, target_pose)
        loss = loss_class + config.BETA * loss_corr

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_class_loss += loss_class.item()
        total_corr_loss += loss_corr.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_corr_loss = total_corr_loss / num_batches
    return avg_loss, avg_class_loss, avg_corr_loss

@torch.no_grad()
def evaluate(model, loader, loss_fn_class, loss_fn_corr, edge_index, device):
    """
    Runs evaluation on the test/validation set and returns losses and classification accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_class_loss = 0.0
    total_corr_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for (input_pose, target_pose, label) in loader:
        input_pose = input_pose.to(device)
        target_pose = target_pose.to(device)
        label = label.to(device)

        B = input_pose.shape[0] # Batch size
        N = input_pose.shape[1] # Number of nodes (57)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N)

        pred_logits, pred_corrected_pose = model(input_pose, edge_index, batch_vec, labels=None, use_feedback=True)

        loss_class = loss_fn_class(pred_logits, label)
        loss_corr = loss_fn_corr(pred_corrected_pose, target_pose)
        loss = loss_class + config.BETA * loss_corr

        _, predicted = torch.max(pred_logits.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        total_loss += loss.item()
        total_class_loss += loss_class.item()
        total_corr_loss += loss_corr.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_corr_loss = total_corr_loss / num_batches
    accuracy = 100 * correct / total if total > 0 else 0.0
    return avg_loss, avg_class_loss, avg_corr_loss, accuracy