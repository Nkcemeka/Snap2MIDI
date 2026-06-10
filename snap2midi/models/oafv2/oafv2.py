import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from nnAudio2.features.mel import MelSpectrogram

class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFramesV2(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        model_complexity = config["model_complexity"]
        input_features = config["in_features"]
        output_features = config["out_features"]

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

        self.config = config
        self.bce = nn.BCELoss()
        self.mel = MelSpectrogram(
            sr=config["sample_rate"], n_fft=config["n_fft"], n_mels=config["n_mels"],\
            hop_length=config["hop_length"], htk=config["htk"], fmin=config["fmin"], \
            fmax=config["fmax"], pad_mode=config["pad_mode"], center=config["center"], \
            window=config["window"]
        )
        self.initialize_weights()
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

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred
    
    def bce_loss(self, preds, target):
        return self.bce(preds, target)
    
    def loss_velocity(self, velocity_pred: torch.Tensor, velocity_label: torch.Tensor, \
                  onset_label: torch.Tensor) -> torch.Tensor:
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['learning_rate_decay_steps'], \
                       gamma=self.config['learning_rate_decay_rate'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def training_step(self, train_batch, batch_idx):
        audio = train_batch["audio"]
        y_onset = train_batch["onset"]
        y_offset = train_batch["offset"]
        y_velocity = train_batch["velocity"]
        y_frame = train_batch["frame"]
        y_velocity = y_velocity.float()

        # Forward pass
        x = self.mel(audio)
        # compute the log of the mel spectrogram
        x = torch.log(torch.clamp(x, min=1e-5)).transpose(-1, -2)
        x = x[:, :y_onset.shape[1], :]

        on_preds, off_preds,  _, frame_preds, vel_preds = self.forward(x)

        # Loss
        onset_loss = self.bce_loss(on_preds, y_onset)
        offset_loss = self.bce_loss(off_preds, y_offset)
        frame_loss = self.bce_loss(frame_preds, y_frame)
        velocity_loss = self.loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + offset_loss + velocity_loss

        self.log_dict({
            'train_onset_loss': onset_loss.item(),
            'train_offset_loss': offset_loss.item(),
            'train_frame_loss': frame_loss.item(),
            'train_velocity_loss': velocity_loss.item(),
            'train_total_loss': loss.item()
        }, logger=True, on_step=False, on_epoch=True)
        return loss
    
    # def training_step(self, train_batch, batch_idx):
    #     x, y_frame, y_onset, y_offset, y_velocity, audio = train_batch
    #     y_velocity = y_velocity.float()

    #     # Forward pass
    #     on_preds, off_preds,  _, frame_preds, vel_preds = self.forward(x)

    #     # Loss
    #     onset_loss = self.bce_loss(on_preds, y_onset)
    #     offset_loss = self.bce_loss(off_preds, y_offset)
    #     frame_loss = self.bce_loss(frame_preds, y_frame)
    #     velocity_loss = self.loss_velocity(vel_preds, y_velocity, y_onset)
    #     loss = onset_loss + frame_loss + offset_loss + velocity_loss

    #     self.log_dict({
    #         'train_onset_loss': onset_loss.item(),
    #         'train_offset_loss': offset_loss.item(),
    #         'train_frame_loss': frame_loss.item(),
    #         'train_velocity_loss': velocity_loss.item(),
    #         'train_total_loss': loss.item()
    #     }, logger=True, on_step=False, on_epoch=True)
    #     return loss

    def validation_step(self, val_batch, batch_idx):
        audio = val_batch["audio"]
        y_onset = val_batch["onset"]
        y_offset = val_batch["offset"]
        y_velocity = val_batch["velocity"]
        y_frame = val_batch["frame"]
        y_velocity = y_velocity.float()

        # Forward pass
        x = self.mel(audio)
        # compute the log of the mel spectrogram
        x = torch.log(torch.clamp(x, min=1e-5)).transpose(-1, -2)
        x = x[:, :y_onset.shape[1], :]

        # Forward pass
        on_preds, off_preds,  _, frame_preds, vel_preds = self.forward(x)

        # Loss
        onset_loss = self.bce_loss(on_preds, y_onset)
        offset_loss = self.bce_loss(off_preds, y_offset)
        frame_loss = self.bce_loss(frame_preds, y_frame)
        velocity_loss = self.loss_velocity(vel_preds, y_velocity, y_onset)
        loss = onset_loss + frame_loss + offset_loss + velocity_loss

        self.log_dict({
            'val_onset_loss': onset_loss.item(),
            'val_offset_loss': offset_loss.item(),
            'val_frame_loss': frame_loss.item(),
            'val_velocity_loss': velocity_loss.item(),
            'val_total_loss': loss.item()
        }, logger=True, on_step=False, on_epoch=True)
        return loss
