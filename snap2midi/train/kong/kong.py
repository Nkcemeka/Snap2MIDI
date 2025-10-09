import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def layer_initialize(layer: nn.Module) -> None:
    """ 
        Initialize Convolutional or Linear layers
        with Glorot initialization

        Args:
            layer (nn.Module): Pytorch layer
    """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.zero_()

def bnorm_initialize(bnorm_layer: nn.Module) -> None:
    """ 
        Initalize a batch norm layer

        Args:
            bnorm_layer: Batch Normalization layer
        
        Returns:
            None
    """
    bnorm_layer.bias.data.fill_(0.)
    bnorm_layer.weight.data.fill_(1.)

def uniform_init(tensor: torch.Tensor):
    """ 
        Performs uniform initialization

        Args:
            tensor (torch.Tensor): Input tensor to initialize
    """
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    lower_bound = - math.sqrt(3.0/fan_in)
    upper_bound = - lower_bound
    nn.init.uniform_(tensor, lower_bound, upper_bound)

def init_gru_weights(tensor: torch.Tensor, funcs: list):
    """
        Init the GRU weights. Pytorch
        groups the weights in the GRU eqns
        (for the input and hidden state) into
        big tensors. So, we need to slice it
        and then initialize each differently.

        Args:
            tensor (torch.Tensor): Input tensor
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

def calc_conv_shape(length: int, padding: int, dilation: int, k_size: int, stride:int) -> int:
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
    
    def forward(self, x: torch.Tensor, kernel_shape: tuple=(1,2)) -> torch.Tensor:
        """
            Forward pass through the convolutional block
            with some pooling. Only average pooling is 
            allowed!

            Args:
                x (torch.Tensor): shape of (batch, in_chans, time_steps, embed_dim)
                kernel_shape (tuple): Kernel shape for pooling
            
            Returns:
                out (torch.Tensor)
        """
        x = self.conv1(x)
        x = self.relu(self.bnorm1(x))
        x = self.conv2(x)
        x = self.relu(self.bnorm2(x))
        x = F.avg_pool2d(x, kernel_size=(kernel_shape))
        return x 
    
class AcousticModel(nn.Module):
    def __init__(self, classes: int, num_features: int, momentum: float, 
                 cmp: int = 48, factors: list=[16, 32, 32]):
        """
            Default constructor for Acoustic model

            Args:
                classes (int): Number of classes for the model
                num_features (int): size of embedding dimension
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
        height = num_features
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

class KongModel(nn.Module):
    def __init__(self, extraction_config, momentum: float, cmp: int=48, 
                 factors: list=[16, 32, 32]):
        """
            Base model for Kong.

            Args:
                momentum (float): Momentum for batch norm layers
                cmp (int): Complexity for Acoustic model
                factors (list): List of factors for the complexity of the conv block
        """
        super().__init__()
        self.num_features = extraction_config["n_mels"]
        self.classes = extraction_config["max_pitch"] - extraction_config["min_pitch"] + 1
        self.momentum = momentum
        sample_rate = extraction_config["sample_rate"]
        window_size = extraction_config["mel_n_fft"]
        hop_size = sample_rate // extraction_config["frame_rate"]
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=self.num_features, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)
    
        self.frame_model = AcousticModel(self.classes, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        self.reg_onset_model = AcousticModel(self.classes, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        self.reg_offset_model = AcousticModel(self.classes, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        self.velocity_model = AcousticModel(self.classes, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        
        self.bn0 = nn.BatchNorm2d(self.num_features, momentum)
        self.reg_onset_gru = nn.GRU(input_size=self.classes*2, hidden_size=256, num_layers=1,
                                    bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, self.classes)
        self.frame_gru = nn.GRU(input_size=self.classes* 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, self.classes, bias=True)

        # initialize the weights for architecture
        self.weights_init()
    
    def weights_init(self):
        """
            Weight initialization for the Kong 
            model!
        """
        bnorm_initialize(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        layer_initialize(self.reg_onset_fc)
        layer_initialize(self.frame_fc)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
            Forward pass for Kong's model

            Args:
                x (torch.Tensor): (batch, samples)

            Returns:
                dict: Dictionary containing the outputs for 
                      reg_onset, reg_offset, frame and velocity
                      rolls.
        """
        # put an assertion to see if there are nan values or infs in the input
        # assert not torch.isnan(x).any(), "NaN values in input"
        # assert not torch.isinf(x).any(), "Inf values in input"
        # print("Min and Max values in input:", x.min().item(), x.max().item())

        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # Put an assertion to see if there are nan values or infs in the output from logmel_extractor
        # assert not torch.isnan(x).any(), "NaN values from log-mel spectrogram"
        # assert not torch.isinf(x).any(), "Inf values from log-mel spectrogram"
        # print("Min and Max values after logmel extraction:", x.min().item(), x.max().item())

        x = x.transpose(1, 3)

        # The transpose above shows Kong did the batch normalization 
        # considering each embedding dimension as a feature: interesting!
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_roll = self.frame_model(x)  # (batch, time_steps, classes)
        reg_onset_roll = self.reg_onset_model(x)  # (batch, time_steps, classes)
        reg_offset_roll = self.reg_offset_model(x)    # (batch, time_steps, classes)
        velocity_roll = self.velocity_model(x)    # (batch, time_steps, classes)
 
        # Use velocities to condition onset regression
        # The velocities were scaled by Kong based on the onsets: interesting stuff
        x = torch.cat((reg_onset_roll, (reg_onset_roll ** 0.5) * velocity_roll.detach()), dim=2)
        (x, _) = self.reg_onset_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        reg_onset_roll = torch.sigmoid(self.reg_onset_fc(x)) #(batch, time_steps, classes)

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_roll, reg_onset_roll.detach(), reg_offset_roll.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_roll = torch.sigmoid(self.frame_fc(x))  # (batch, time_steps, classes)

        # print the min and max values of the outputs to check for anomalies
        # print("Min and Max values after models:")
        # print("Reg Onset Roll - Min:", reg_onset_roll.min().item(), "Max:", reg_onset_roll.max().item())
        # print("Reg Offset Roll - Min:", reg_offset_roll.min().item(), "Max:", reg_offset_roll.max().item())
        # print("Frame Roll - Min:", frame_roll.min().item(), "Max:", frame_roll.max().item())
        # print("Velocity Roll - Min:", velocity_roll.min().item(), "Max:", velocity_roll.max().item())

        return {
            'reg_onset_roll': reg_onset_roll, 
            'reg_offset_roll': reg_offset_roll, 
            'frame_roll': frame_roll, 
            'velocity_roll': velocity_roll
        }

class KongPedal(nn.Module):
    def __init__(self, extraction_config: dict, momentum: float, cmp: int=48, 
                 factors: list=[16, 32, 32]):
        """
            Base model for Kong.

            Args:
                num_features (int): Size of embedding dimension
                momentum (float): Momentum for batch norm layers
                cmp (int): Complexity for Acoustic model
                factors (list): List of factors for the complexity of the conv block
        """
        super().__init__()
        self.num_features = extraction_config["n_mels"]
        self.momentum = momentum
        sample_rate = extraction_config["sample_rate"]
        window_size = extraction_config["mel_n_fft"]
        hop_size = sample_rate // extraction_config["frame_rate"]
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
            hop_length=hop_size, win_length=window_size, window=window, 
            center=center, pad_mode=pad_mode, freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
            n_fft=window_size, n_mels=self.num_features, fmin=fmin, fmax=fmax, ref=ref, 
            amin=amin, top_db=top_db, freeze_parameters=True)
        
        self.reg_pedal_onset_model = AcousticModel(1, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        self.reg_pedal_offset_model = AcousticModel(1, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        self.reg_pedal_frame_model = AcousticModel(1, self.num_features, self.momentum,
                                         cmp=cmp, factors=factors)
        
        self.bn0 = nn.BatchNorm2d(self.num_features, momentum)

        # initialize the weights for architecture
        self.weights_init()
    
    def weights_init(self):
        """
            Weight initialization for the 
            architecture!
        """
        bnorm_initialize(self.bn0)
    

    def forward(self, x: torch.Tensor) -> dict:
        """
            Forward pass for Kong's Pedal model

            Args:
                x (torch.Tensor): (batch, samples)

            Returns:
                dict: Dictionary containing the rolls for 
                      reg_pedal_onset_roll, reg_pedal_offset_roll, 
                      and pedal_frame_roll.
        """
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)

        # The transpose above shows Kong did the batch normalization 
        # considering each embedding dimension as a feature: interesting!
        x = self.bn0(x)
        x = x.transpose(1, 3)

        reg_pedal_onset_roll = self.reg_pedal_onset_model(x)  # (batch, time_steps, 1)
        reg_pedal_offset_roll = self.reg_pedal_offset_model(x)  # (batch, time_steps, 1)
        pedal_frame_roll = self.reg_pedal_frame_model(x)  # (batch, time_steps, 1)
        
        return {
            'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll,
            'pedal_frame_roll': pedal_frame_roll
        }

# Test Kong's pedal model
if __name__ == "__main__":
    model = KongPedal(num_features=229, momentum=0.1)
    x = torch.randn(4, 640, 229)  # (batch, time_steps, embed_dim)
    out = model(x)

    for key in out:
        print(f"{key}: {out[key].shape}")

    print("Kong's pedal model initialized and tested successfully!")
