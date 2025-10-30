# main.py
# Main script to run the training and evaluation process.

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import datetime

import config
import utils
from data_loader import create_dataloaders
from model import DualBranchGCN, create_skeleton_graph
from train import train_one_epoch, evaluate

def main():
    # --- Setup ---
    utils.set_seed(42)

    log_dir = os.path.join(config.SAVE_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.get_logger(os.path.join(log_dir, "run.log"))
    logger.info(f"Using device: {config.DEVICE}")

    # --- Data ---
    logger.info("Loading data...")
    train_loader, test_loader = create_dataloaders(config.DATA_PATH)
    logger.info("Data loaded.")

    # --- Model ---
    logger.info("Building model...")
    edge_index = create_skeleton_graph().to(config.DEVICE)

    model = DualBranchGCN(
        in_channels=config.NUM_FRAMES,
        hidden_channels=config.HIDDEN_DIM,
        out_channels_class=config.NUM_CLASSES,  # Now in config.py
        out_channels_corr=config.NUM_FRAMES,  # Output should match input frame count
        num_blocks=config.NUM_GCN_BLOCKS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)

    # --- Training Setup ---
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.LR_DECAY_STEP, 
        gamma=config.LR_DECAY_GAMMA
    )

    # Loss functions
    loss_fn_class = nn.CrossEntropyLoss().to(config.DEVICE)
    loss_fn_corr = nn.MSELoss().to(config.DEVICE)

    # --- Main Loop ---
    logger.info("Starting training...")
    best_accuracy = 0.0
    epoch_iter = tqdm(range(config.EPOCHS), desc="Total Progress")
    for epoch in epoch_iter:

        train_loss, tr_class, tr_corr = train_one_epoch(
            model, train_loader, optimizer, loss_fn_class, loss_fn_corr, edge_index, config.DEVICE
        )
        val_loss, val_class, val_corr, accuracy = evaluate(
            model, test_loader, loss_fn_class, loss_fn_corr, edge_index, config.DEVICE
        )
        scheduler.step()

        epoch_iter.set_postfix(
            Train_Loss=f"{train_loss:.4f}",
            Val_Loss=f"{val_loss:.4f}",
            Val_Acc=f"{accuracy:.2f}%"
        )

        logger.info(
            f"Epoch {epoch+1}/{config.EPOCHS} | Val Acc: {accuracy:.2f}% | Best Acc: {best_accuracy:.2f}%"
        )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            logger.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    logger.info("Training complete.")
    logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()