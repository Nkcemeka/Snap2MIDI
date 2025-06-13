"""
File: conv_shallow.py
Author: Chukwuemeka L. Nkama
Date: 6/12/2025

Description: A convolutional shallow neural network performing multi-pitch
             estimation/transcription!
"""

# Imports
import torch
from torch import nn

def calc_conv_shape(length: int, padding: int, dilation: int, k_size: int, stride:int):
    """
        Calculate shape of tensor from a
        conv. block

        Args:
        -----
            length (int): Length of the input tensor
            padding (int): Padding applied to the input tensor
            dilation (int): Dilation applied to the input tensor
            k_size (int): Kernel size of the conv. block
            stride (int): Stride of the conv. block
        
        Returns:
        -------
            int: Shape of the tensor after the conv. block
    """
    return (length + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1

class ConvShallowTranscriber(nn.Module):
    def __init__(self, \
            in_features: int,
            out_features: int = 128,
            dropout: float = 0.2) -> None:
        """
            Default constructor for ConvShallowTranscriber. 
            Has a smaller number of params compared to its
            shallow transcriber counterpart.

            Args:
            -----
                in_features (int): Embedding dim. of input representation
                out_features(int): Typically, 128 (range of MIDI pitches)
                dropout (float): Dropout rate

            Returns:
            -------
                None
        """
        super().__init__()
        self.in_features = in_features
        self.dropout = dropout 
        self.out_features = out_features

        # create conv shallow neural network
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=(3,3), padding=1)
        self.drop = nn.Dropout(self.dropout)

        final_embedding_length = self.in_features
        for _ in range(2):
            # calculate the shape of the tensor after each conv block
            final_embedding_length = calc_conv_shape(final_embedding_length, 1, 1, 3, 1)
        
        final_embedding_length *= 4 # 4 is the number of channels after conv2
        self.fc = nn.Linear(final_embedding_length, self.out_features)
        # initialize weights
        self.weight_init()

    def weight_init(self):
        """
            Initialize the model weights after
            class has been instantiated!
        """
        for each in self.modules():
            if isinstance(each, nn.Linear):
                # Initialize weights using He Initialization
                # a is 0 by default in torch, so leaky relu becomes relu
                nn.init.kaiming_normal_(each.weight) 
                if each.bias is not None:
                    each.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass through the shallow 
            transcriber

            Args:
            -----
                x (torch.Tensor): Input mid-level representation
                                  Shape of (Batch size, Frames, in_features)

            Returns:
            -------
                out (torch.Tensor): Predictions of shape (frames, out_features)
        """
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        batch_size, chan, time_steps, embed_dim = x.shape
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = out.transpose(1,2).flatten(-2)
        out = self.fc(out)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
            Return a piano roll given input representation.
            
            Args:
            -----
                x (torch.Tensor): Input mid-level representation

            Returns:
            -------
                out (torch.Tensor): Piano roll of shape (frames, out_features)
        """
        return torch.sigmoid(self.forward(x))
