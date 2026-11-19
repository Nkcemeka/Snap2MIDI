import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class AcousticModel(nn.Module):
    """ 
        This is a convolutional stack that
        learns acoustic features based on
        the input representation.
    """
    def __init__(self, params: dict):
        """ 
            Instantiate Acoustic Model class
        
            Args
            -----
                params (dict): Model parameters
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
            Forward pass.

            Args
            -----
                x (torch.Tensor): Input representation
            
            Returns
            -------
                x (torch.Tensor): Processed representation
        """
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
    # cuDNN holds the RNN sequence length in a 16-bit field, so one call with
    # 65,536 timesteps or more fails with CUDNN_STATUS_NOT_SUPPORTED -- not an
    # out-of-memory, a hard ceiling (measured: 65,535 passes, 65,536 does not).
    # Training never approaches it, but the cross-dataset benchmark scores whole
    # pieces: 15 of Studio MAESTRO's 1,276 run past it, the longest 82,127
    # frames (44 minutes at 31.25 fps).
    cudnn_max_seq_length = 65535
    inference_chunk_length = 512

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        Default constructor for BiLSTM

        Args
        ----
            input_size (int): Number of input features
            hidden_size (int): Number of hidden features
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
            Forward pass.

            Args
            -----
                x (torch.Tensor): Input representation
            
            Returns
            -------
                x (torch.Tensor): Processed representation
        """
        if x.shape[1] > self.cudnn_max_seq_length:
            return self._chunked(x)
        x, _ = self.lstm(x)
        return x

    def _chunked(self, x: torch.Tensor) -> torch.Tensor:
        """The same result as one call, in chunks cuDNN will accept.

        The hidden and cell state are carried across chunk boundaries -- forward
        left to right, backward right to left -- so every timestep still sees the
        whole sequence and this is the full-sequence recurrence, not a windowed
        approximation of it. It is the BiLSTM that `models/oafv2/oafv2.py` and
        `models/hpp/nets.py` already run in eval mode, applied here only above
        the ceiling so that every sequence the single call can handle keeps
        taking it and no existing result moves.
        """
        batch_size, sequence_length, _ = x.shape
        hidden_size = self.lstm.hidden_size
        num_directions = 2 if self.lstm.bidirectional else 1

        h = torch.zeros(num_directions, batch_size, hidden_size,
                        device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        output = torch.zeros(batch_size, sequence_length,
                             num_directions * hidden_size,
                             device=x.device, dtype=x.dtype)

        slices = range(0, sequence_length, self.inference_chunk_length)
        for start in slices:
            end = start + self.inference_chunk_length
            output[:, start:end, :], (h, c) = self.lstm(x[:, start:end, :], (h, c))

        if self.lstm.bidirectional:
            # The backward half written above ran left to right and is wrong;
            # redo it right to left, which is the direction it belongs to.
            h.zero_()
            c.zero_()
            for start in reversed(slices):
                end = start + self.inference_chunk_length
                result, (h, c) = self.lstm(x[:, start:end, :], (h, c))
                output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

        return output


class OnsetsAndFrames(pl.LightningModule):
    def __init__(self, params: dict):
        """ 
            Instantiate OnsetsAndFrames

            Args
            ----
                params (dict): Configuration parameters and other info.
        """
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

        self.config = params
        self.bce = nn.BCEWithLogitsLoss()
        self.initialize_weights()
        self.automatic_optimization = False
        self.save_hyperparameters()
    
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
        """ 
            Forward pass.

            Args
            -----
                x (torch.Tensor): Input representation
            
            Returns
            -------
                onset_pred (torch.Tensor): Onset predictions
                activation_pred (torch.Tensor): Activation predictions
                frame_pred (torch.Tensor): Frame-level predictions
                velocity_pred (torch.Tensor): Velocity predictions
        """
        onset_pred = self.onset_stack(x)
        activation_pred = self.frame_stack(x)
        combined_pred = torch.cat([torch.sigmoid(onset_pred).detach(), \
                    torch.sigmoid(activation_pred)], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(x)
        return onset_pred, activation_pred, frame_pred, velocity_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, \
                    gamma=self.config['learning_rate_decay_rate'])

        return{
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
    def bce_loss(self, preds, target):
        return self.bce(preds, target)
    
    def weighted_bce_loss(self, output, target, weights):
        loss_matrix = F.binary_cross_entropy_with_logits(output, target, reduction='none')
        weighted_loss = loss_matrix * weights
        return weighted_loss.mean()
    
    def loss_velocity(self, velocity_pred: torch.Tensor, velocity_label: torch.Tensor, \
                  onset_label: torch.Tensor) -> torch.Tensor:
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        x, y_frame, y_onset, y_velocity, label_weights, audio = train_batch
        y_velocity = y_velocity.float()
        label_weights = label_weights.float()

        # Forward pass
        on_preds, _, frame_preds, vel_preds = self.forward(x)

        # Loss
        onset_loss = self.bce_loss(on_preds, y_onset)
        frame_loss = self.weighted_bce_loss(frame_preds, y_frame, label_weights)
        velocity_loss = self.loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + velocity_loss

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)

        self.clip_gradients(
            optimizer,
            gradient_clip_algorithm="norm",
            gradient_clip_val=self.config["clip_gradient_norm"]
        )

        optimizer.step()

        if (self.global_step) % self.config["learning_rate_decay_steps"] == 0:
            scheduler.step()

        self.log_dict({
            'train_onset_loss': onset_loss,
            'train_frame_loss': frame_loss,
            'train_velocity_loss': velocity_loss,
            'train_total_loss': loss
        }, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y_frame, y_onset, y_velocity, label_weights, audio = val_batch
        y_velocity = y_velocity.float()
        label_weights = label_weights.float()

        # Forward pass
        on_preds, _, frame_preds, vel_preds = self.forward(x)

        # Loss
        onset_loss = self.bce_loss(on_preds, y_onset)
        frame_loss = self.weighted_bce_loss(frame_preds, y_frame, label_weights)
        velocity_loss = self.loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + velocity_loss

        self.log_dict({
            'val_onset_loss': onset_loss,
            'val_frame_loss': frame_loss,
            'val_velocity_loss': velocity_loss,
            'val_total_loss': loss
        }, logger=True, on_step=False, on_epoch=True)
        return loss
