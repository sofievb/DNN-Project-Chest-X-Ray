import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

def train(model, device, loader, optimizer, criterion, scheduler=None, epochs=100, patience=25, early_stop_patience=False):
    train_losses = []
    train_accuracies = []
    correct_train = 0
    

    if early_stop_patience:
        early_stop_patience = patience
        epochs_no_improve = 0

    best_validation_loss = float('inf')

    for epoch in trange(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in loader:
            #if gpu is available, move to device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_train_loss / len(loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}\n Training Loss: {avg_train_loss:.4f} Training Accuracy: {train_accuracy:.2f}%")

        # Early stopping
        if avg_train_loss < best_validation_loss:
            best_validation_loss = avg_train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stop_patience and epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch+1}\n Training Loss: {avg_train_loss:.4f} Training Accuracy: {train_accuracy:.2f}%,")
    return train_losses,train_accuracies

def validation(model, scheduler, device, val_loader, criterion):
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    y_true = []
    y_pred = []
    val_losses = []
    val_accuracies = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    return val_losses, val_accuracies