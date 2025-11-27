import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_device():
    # Use MPS on your Mac if available, otherwise fall back to CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(weights_path: str = "model_weights.pth") -> tuple[nn.Module, torch.device]:
    device = get_device()
    # map_location="cpu" is safe; weights are device-agnostic
    state_dict = torch.load(weights_path, map_location="cpu")

    model = SimpleMLP()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device
