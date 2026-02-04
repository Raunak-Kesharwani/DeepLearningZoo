import torch

def load_torchscript_model(model_path: str):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model
