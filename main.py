import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.data_loader import DataProcessing
from ResNet.resnet50 import ResNet
from ResNet.train import train, validation
from torchvision import transforms as tf
import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accuracies, val_losses):
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Plot training accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()
    plt.show()

CONFIG = {
    "img_dir": "path/to/images",
    "annot_file": "path/to/annotations.csv",
    "train_txt": "path/to/train.txt",
    "test_txt": "path/to/test.txt",
    "multi_label": True,  # Set to False for single-label classification
    "batch_size": 32,
    "val_split": 0.2,
    "n_workers": 4,
    "num_classes": 14,  # Adjust based on the dataset
    "learning_rate": 0.001,
    "epochs": 25,
    "patience": 5,
    "debug": False,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def main():
    # Data transforms
    transform = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data loading
    data_processor = DataProcessing(
        img_dirs=CONFIG["img_dir"],
        annot_file=CONFIG["annot_file"],
        transform=transform,
        multi_label=CONFIG["multi_label"]
    )

    train_loader, val_loader = data_processor.load_train_val_data(
        txt_file=CONFIG["train_txt"],
        batchsize=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        n_workers=CONFIG["n_workers"]
    )

    # Model
    model = ResNet(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])

    # Loss function
    if CONFIG["multi_label"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training
    print("Starting training...")
    train_losses, train_accuracies, val_losses = train(
        model=model,
        device=CONFIG["device"],
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=CONFIG["epochs"],
        patience=CONFIG["patience"],
        multi_label=CONFIG["multi_label"],
        debug=CONFIG["debug"],
        val_loader=val_loader
    )

    # Plot metrics
    plot_metrics(train_losses, train_accuracies, val_losses)

if __name__ == "__main__":
    main()
