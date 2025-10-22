import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticModel(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        temporal_sizes = params.get("temporal_sizes", [3, 3, 3])
        freq_sizes = params.get("freq_sizes", [3, 3, 3])
        out_channels = params.get("out_channels", [32, 32, 64])
        pool_sizes = params.get("pool_sizes", [1, 2, 2])
        dropout_probs = params.get("dropout_probs", [0, 0.25, 0.25])
        dropout_fc = params.get("dropout_fc", 0.5)
        fc_size = params.get("fc_size", 512)

        assert len(temporal_sizes) == len(freq_sizes) == len(out_channels) == len(pool_sizes) == len(dropout_probs), \
            "All parameter lists must have the same length"
        
        self.conv_blocks = nn.ModuleList()
        num_blocks = len(temporal_sizes)

        for i in range(num_blocks):
            in_channels = 1 if i == 0 else out_channels[i - 1]
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels[i], kernel_size=(temporal_sizes[i], freq_sizes[i]), padding="same"),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pool_sizes[i])) if pool_sizes[i] > 1 else nn.Identity(),
                nn.Dropout(p=dropout_probs[i]) if dropout_probs[i] > 0 else nn.Identity()
            )
            self.conv_blocks.append(conv_block)
            
        self.fc = nn.Sequential(
            nn.Linear(in_features=out_channels[-1]*(params.get("in_features", 229)//4), out_features=fc_size),
            nn.Dropout(p=dropout_fc),
        )

    def forward(self, x):
        # x is the mel spectrogram or some embedding
        # of shape (B, t, d) where B is the batch size,
        # t is the time dimension, and d is the feature dimension
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2]) # Reshape to (B, 1, t, d)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
    

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Default constructor for BiLSTM

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden features
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, params: dict):
        # JongWook's implementation of Onsets and Frames
        # has 26M parameters but it should be smaller if 
        # we follow the original paper strictly
        # It should be about 6,796,096 parameters which is approx
        # 6.8M parameters
        super().__init__()
        out_features = params.get("out_features", 88)
        fc_size = params.get("fc_size", 512)
        onset_lstm_units = params.get("onset_lstm_units", 128)
        combined_lstm_units = params.get("combined_lstm_units", 128)

        self.onset_stack = nn.Sequential(
            AcousticModel(params),
            BiLSTM(input_size=fc_size, hidden_size=onset_lstm_units),
            nn.Linear(onset_lstm_units*2, out_features),
        )

        self.frame_stack = nn.Sequential(
            AcousticModel(params),
            nn.Linear(fc_size, out_features),
        )

        self.combined_stack = nn.Sequential(
            BiLSTM(input_size=out_features*2, hidden_size=combined_lstm_units),
            nn.Linear(combined_lstm_units*2, out_features),
        )

        self.velocity_stack = nn.Sequential(
            AcousticModel(params),
            nn.Linear(fc_size, out_features),
        )
    
    def initialize_weights(self):
        """Mimic TF-Slim variance scaling init (factor=2, fan_avg) + bias init."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        onset_pred = self.onset_stack(x)
        activation_pred = self.frame_stack(x)
        combined_pred = torch.cat([torch.sigmoid(onset_pred).detach(), \
                    torch.sigmoid(activation_pred)], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(x)
        return onset_pred, activation_pred, frame_pred, velocity_pred
