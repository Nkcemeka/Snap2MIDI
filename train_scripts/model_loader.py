"""
    File: model_loader.py
    Author: Chukwuemeka L. Nkama
    Date: 4/4/2025
    Description: Instnatiates models
"""

# Imports
import torch
from models import *

def load_shallow(in_features, hidden_units, dropout):
    model = ShallowTranscriber(in_features, hidden_units, \
            dropout=dropout)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    return model

