import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from utils.metrics import Metrics
from torch.utils.tensorboard import SummaryWriter

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Train Function
def train(model, device, train_loader, optimizer, criterion, scheduler=None, epochs=100, patience=25, multi_label=False, debug=False, val_loader=None):
    writer = SummaryWriter()
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    train_accuracies = []
    val_losses = []

    best_validation_loss = float('inf')

    for epoch in trange(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            if debug and i > 0:
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if multi_label:
                loss = criterion(outputs, labels.float())
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == labels).all(dim=1).sum().item()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            total_train += labels.size(0)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Logging to TensorBoard
        writer.add_scalar('Training Loss', avg_train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)

        print(f"Epoch {epoch + 1}: Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Run validation at the end of each epoch
        if val_loader is not None:
            avg_val_loss = validation(model, device, val_loader, criterion, multi_label=multi_label)
            val_losses.append(avg_val_loss)

            # Logging to TensorBoard
            writer.add_scalar('Validation Loss', avg_val_loss, epoch)

            # Update scheduler based on validation loss
            if scheduler:
                scheduler.step(avg_val_loss)

            # Early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Save model if validation loss has improved
            if avg_val_loss < best_validation_loss:
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Model improved and saved at epoch {epoch + 1}")
                best_validation_loss = avg_val_loss

        if debug:
            print("Debug mode - stopping after one epoch.")
            break

    writer.close()
    return train_losses, train_accuracies, val_losses

def validation(model, device, val_loader, criterion, multi_label=False):
    model.eval()  # Sett modellen til evalueringsmodus
    running_val_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():  # Ingen gradientberegninger under validering
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if multi_label:
                val_loss = criterion(outputs, labels.float())
                predictions = (outputs > 0.5).float().cpu().numpy()  # Konverter til binære prediksjoner
                y_pred.extend(predictions)
            else:
                val_loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)
                y_pred.extend(predictions.cpu().numpy())

            running_val_loss += val_loss.item()
            y_true.extend(labels.cpu().numpy())

    avg_val_loss = running_val_loss / len(val_loader)

    # Beregn metrikker
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("Validation Metrics:")
    print(Metrics.calculate_all(y_true, y_pred, multi_label=multi_label))

    return avg_val_loss
