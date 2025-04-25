"""
File: onsets_and_frames.py
Author: Chukwuemeka L. Nkama
Date: 4/11/2025
Credits: This implementation was gotten from the 
         Onsets and Frames model from:
         https://github.com/jongwook/onsets-and-frames
Description: Implementation of the Onsets and Frames model 
             for music transcription.
"""

# Imports
import torch
from torch import nn
from typing import List, Optional, Tuple
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Default constructor for BiLSTM
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden features
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features, factor=16):
        """
        Default constructor for ConvStack
        Args:
            input_features (int): Number of input features
            output_features (int): Number of output features
            factor (int): Factor for the number of channels
        """
        super().__init__()

        # Enusre factor is a power of 2
        if factor & (factor - 1) != 0:
            # bitwise check to see if factor is a power of 2
            # 0b11111111 & 0b11111110 = 0b11111110
            raise ValueError(f"Factor {factor} is not a power of 2!")

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // factor, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // factor),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // factor, output_features // factor, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // factor),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // factor, output_features // int(factor/2), (3, 3), padding=1),
            nn.BatchNorm2d(output_features // int(factor/2)),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // int(factor/2)) * (input_features // int(factor/4)), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, factor=16, model_complexity=48):
        super().__init__()

        model_size = model_complexity * factor
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, x):
        onset_pred = self.onset_stack(x)
        offset_pred = self.offset_stack(x)
        activation_pred = self.frame_stack(x)
        combined_pred = torch.cat([torch.sigmoid(onset_pred).detach(), \
                    torch.sigmoid(offset_pred).detach(), \
                    torch.sigmoid(activation_pred)], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(x)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
