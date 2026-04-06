import torch
import os
from hidden_state_detection.models import MLPNet


def load_mlp_by_weights(gpu_device, weight_path, dropout=0.5):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPNet(dropout=dropout).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model
