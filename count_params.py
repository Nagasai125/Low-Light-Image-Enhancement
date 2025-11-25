import torch
from model import MainNetwork
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    model = MainNetwork()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
    print(f"IlluminationEstimator: {sum(p.numel() for p in model.enhance.parameters())}")
    print(f"FirstStageDenoiser: {sum(p.numel() for p in model.denoise_1.parameters())}")
    print(f"SecondStageDenoiser: {sum(p.numel() for p in model.denoise_2.parameters())}")

if __name__ == "__main__":
    main()
