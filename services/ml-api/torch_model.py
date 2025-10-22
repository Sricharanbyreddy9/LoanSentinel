import torch
import torch.nn as nn

class RiskNet(nn.Module):
    def __init__(self, in_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

def load_torch_model(weights_path: str, device: str = "cpu"):
    m = RiskNet(in_dim=5, hidden=32)
    state = torch.load(weights_path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m
