"""
File: shallow_network.py
Author: Chukwuemeka L. Nkama
Date: 4/1/2025

Description: A shallow neural network performing multi-pitch
             estimation/transcription!
"""

# Imports
import torch
from torch import nn
from typing import List, Optional, Tuple

# Create class for shallow neural network
class ShallowTranscriber(nn.Module):
    def __init__(self, \
            in_features: int,
            hidden_units: int,
            out_features: int = 128,
            dropout: float = 0.2) -> None:
        """
            Default constructor for ShallowTranscriber

            Args:
                in_features (int): Embedding dim. of input representation
                hidden_units (int): Number of hidden units in the shallow 
                                    transcriber
                out_features(int): Typically, 128 (range of MIDI pitches)
                dropout (float): Dropout rate

            Returns:
                None
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.dropout = dropout 
        self.out_features = out_features

        # create shallow neural network
        self.scn = nn.Sequential(
            nn.Linear(in_features=self.in_features, \
                    out_features = self.hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features = self.hidden_units, \
                    out_features = self.out_features)
        )

        # initialize weights
        self.weight_init()

    def weight_init(self):
        """
            Initialize the model weights after
            class has been instantiated!
        """
        for each in self.modules():
            if isinstance(each, nn.Linear):
                nn.init.xavier_uniform_(each.weight, \
                        nn.init.calculate_gain('relu'))
                if each.bias is not None:
                    each.bias.data.zero_()

    def forward(self, x):
        """
            Forward pass through the shallow 
            transcriber

            Args:
                x (torch.tensor): Input mid-level representation
                                  Shape of (Batch size, Frames, in_features)

            Returns:
                out (torch.tensor): Predictions of shape (frames, out_features)
        """
        out = self.scn(x)
        return out

    def predict(self, x):
        """
            Return a piano roll given input rep. x!
            We assume a Bernoulli prob. distribution
            for each midi pitch and perform a sigmoid
            so we can know what notes are active.

            Hence, the loss function would be a sum of the
            binary cross-entropy loss across all midi pitches
            and across the entire batch under consideration!

            See UDL by Simon Prince for more details!!!
            
            Args:
                x (torch.tensor): Input mid-levl rep.

            Returns:
                out (torch.tensor): Piano roll of shape (frames, out_features)
        """
        return torch.sigmoid(self.forward(x))

