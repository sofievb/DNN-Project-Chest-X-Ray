import torch

def save_model(model, path):
    """
    Saves a PyTorch model to the specified path.
    Args:
        model: The PyTorch model to save.
        path: File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load model from  specified path.
    Args:
        model: PyTorch model architecture to load weights into.
        path: file path to load model into
        device: The device to load the model onto.
    Returns:
        The model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model
