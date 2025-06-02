import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_initialize(layer):
    """ 
        Initialize Convolutional or Linear layers
        with Glorot initialization

        Args:
            layer: Pytorch layer
    """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.zero_()

def bnorm_initialize(bnorm_layer):
    """ 
        Initalize a batch norm layer

        Args:
            bnorm_layer: Batch Normalization
            layer
    """
    bnorm_layer.bias.data.fill_(0.)
    bnorm_layer.weight.data.fill_(1.)

def uniform_init(tensor: torch.Tensor):
    """ 
        Performs uniform initialization

        Args:
            tensor (torch.Tensor)
    """
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    lower_bound = - math.sqrt(3.0/fan_in)
    upper_bound = - lower_bound
    nn.init.uniform_(tensor, lower_bound, upper_bound)

def init_gru_weights(tensor: torch.Tensor, funcs):
    """
        Init the GRU weights. Pytorch
        groups the weights in the GRU eqns
        (for the input and hidden state) into
        big tensors. So, we need to slice it
        and then initialize each differently.

        Args:
            tensor (torch.Tensor)
            funcs: functions for weight init.
    """
    chunk_size = tensor.shape[0]//len(funcs)
    for i, func in enumerate(funcs):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        func(tensor[start:end])

def init_gru(gru):
    for i in range(gru.num_layers):
        # Input to hidden weights
        # GRU splits weight_ih_l{i} into weights for the reset
        # update and candidate hidden state
        init_gru_weights(
            getattr(gru, f'weight_ih_l{i}'),
            [uniform_init, uniform_init, uniform_init]
        )

        # zero all the biases for input to hidden
        torch.nn.init.constant_(getattr(gru, f'bias_ih_l{i}'), 0)

        # Hidden to hidden weights
        # Weights for the reset, update and candidate hidden state
        # applied to the previous hidden state
        init_gru_weights(
            getattr(gru, f'weight_hh_l{i}'),
            [uniform_init, uniform_init, nn.init.orthogonal_]
        )

        # zero all the biases for hidden to hidden
        torch.nn.init.constant_(getattr(gru, f'bias_hh_l{i}'), 0)

def calc_conv_shape(length: int, padding: int, dilation: int, k_size: int, stride:int):
    """
        Calculate shape of tensor from a
        conv. block

        Args:
            length (int): Length of the input tensor
            padding (int): Padding applied to the input tensor
            dilation (int): Dilation applied to the input tensor
            k_size (int): Kernel size of the conv. block
            stride (int): Stride of the conv. block
        
        Returns:
            int: Shape of the tensor after the conv. block
    """
    return (length + 2 * padding - dilation * (k_size - 1) - 1) // stride + 1

class ConvBlock(nn.Module):
    """ 
        Conv Block for the Kong model
        architecture.
    """
    def __init__(self, input_chans: int, output_chans: int, momentum: float):
        """
            Default constructor.

            Args:
                input_chans (int): Input channels for the conv. block
                output_chans (int): Output channels for the conv. block
                momentum (float): Momentum for batch norm (used as a sort of
                                  runing avg for calcs.)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_chans,
            out_channels=output_chans,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels=output_chans,
            out_channels=output_chans,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )

        self.bnorm1 = nn.BatchNorm2d(output_chans, momentum)
        self.bnorm2 = nn.BatchNorm2d(output_chans, momentum)
        self.relu = nn.ReLU()

        # Initialize the weights
        self.weights_init()
    
    def weights_init(self):
        """
            Initialize the convolutional and 
            batch norm layers
        """
        for each in [self.conv1, self.conv2]:
            layer_initialize(each)
        
        for each in [self.bnorm1, self.bnorm2]:
            bnorm_initialize(each)
    
    def forward(self, x: torch.Tensor, kernel_shape: tuple=(1,2)):
        """
            Forward pass through the convolutional block
            with some pooling. Only average pooling is 
            allowed!

            Args:
                x (torch.Tensor): shape of (batch, in_chans, time_steps, embed_dim)
                kernel_shape (tuple): Kernel shape for pooling
            
            Returns:
                out: (batch, out_chans, )
        """
        x = self.conv1(x)
        x = self.relu(self.bnorm1(x))
        x = self.conv2(x)
        x = self.relu(self.bnorm2(x))
        x = F.avg_pool2d(x, kernel_size=(kernel_shape))
        return x 
    
class AcousticModel(nn.Module):
    def __init__(self, classes: int, clue: int, momentum: float, 
                 cmp: int = 48, factors: list=[16, 32, 32]):
        """
            Default constructor for Acoustic model

            Args:
                classes (int): Number of classes for the model
                clue (int): Clue for the model which is size of embedding dimension
                momentum (float): Momentum for batch norm
                cmp (int): Complexity for the model
                factors (list): To vary the complexity of the conv, model
        """
        super().__init__()

        if factors[-1] > 128 or factors[0] < 24:
            cmp = 48
            factors = [16, 32, 32]

        # These conv blocks actually preserve the shape of the embedding dimensions
        # check with calc_conv_shape. However, the pooling is what does the reduction
        self.conv1 = ConvBlock(input_chans=1, 
                    output_chans=cmp, momentum=momentum)
        self.conv2 = ConvBlock(input_chans=cmp, 
                               output_chans=cmp+factors[0], momentum=momentum)
        self.conv3 = ConvBlock(input_chans=cmp+factors[0], 
                               output_chans=cmp+factors[0]+factors[1], momentum=momentum)
        self.conv4 = ConvBlock(input_chans=cmp+factors[0]+factors[1], 
                               output_chans=cmp+factors[0]+factors[1]+factors[2], momentum=momentum)
        
        self.midfeat = cmp + factors[0] + factors[1] + factors[2]

        # Since we know we will pool by a factor of 2, let's cheat
        # a little bit to make the model modular in terms of
        # self.midfeat.
        height = clue
        for _ in range(4):
            height = height // 2
        
        self.midfeat = self.midfeat * height

        self.fc5 = nn.Linear(self.midfeat, 768, bias=False)
        self.bnorm5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc = nn.Linear(512, classes)

        # Initialize the weights of the model
        self.weights_init()
    
    def weights_init(self):
        """
            Initialize the weights of the model
        """
        layer_initialize(self.fc5)
        bnorm_initialize(self.bnorm5)
        init_gru(self.gru)
        layer_initialize(self.fc)
    
    def forward(self, x: torch.Tensor):
        """
            Forward pass for the Acoustic Model.

            Args:
                x (torch.Tensor): (batch, chans, time, embed_dim)
            
            Returns:
                out (torch.Tensor): (batch, time, classes)
        """
        x = self.conv1(x, kernel_shape=(1,2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, kernel_shape=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, kernel_shape=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, kernel_shape=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bnorm5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training)
        out = torch.sigmoid(self.fc(x))
        return out

class KongBaseModel(nn.Module):
    pass


# Test acoustic model
# if __name__ == "__main__":
#     model = AcousticModel(classes=10, clue=229, momentum=0.1)
#     x = torch.randn(32, 1, 100, 229)  # (batch, chans, time, embed_dim)
#     out = model(x)
#     print(out.shape)  # Should be (32, time_steps, classes)