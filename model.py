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
    def forward(self,x):
        return self.net(x)

# def load_model (weights_path: str = "model/weigths/model.pth") -> nn.Module:
#      """
#     Loads the model weights from a .pth file and returns a ready-to-use model in eval mode.
#     We’ll keep inference on CPU for simplicity.
#     """
#     device = torch.device("cpu")  
#     model = SimpleMLP()
#     state_dict = torch.load(weights_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model

def load_model(weights_path: str = "model_weights.pth") -> nn.Module:
    """
    Loads the model weights from a .pth file and returns a ready-to-use model in eval mode.
    We’ll keep inference on CPU for simplicity.
    """
    device = torch.device("mps")  
    model = SimpleMLP()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model