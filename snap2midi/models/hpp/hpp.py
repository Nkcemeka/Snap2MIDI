import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from nnAudio2.Spectrogram import CQT
from .nets import CNNTrunk, FreqGroupLSTM
import torchaudio

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.squeeze(x, self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

class Head(nn.Module):
    def __init__(self, model_size) -> None:
        super().__init__()
        self.head = FreqGroupLSTM(model_size, 1, model_size)
    def forward(self, x):
        # input: [B x model_size x T x 88]
        # output: [B x 1 x T x 88]
        y = self.head(x)
        return y

class SubNet(nn.Module):
    def __init__(self, model_size = 128,  head_names = ['head'], concat = False, time_pooling = False) -> None:
        super().__init__()
        # Trunk
        self.trunk = CNNTrunk(c_in=1, c_har=16, embedding=model_size)

        # Heads
        head_size = model_size
        self.concat = concat
        if(concat):
            head_size *= 2
        self.head_names = head_names
        self.heads = nn.ModuleDict()
        for name in head_names:
            self.heads[name] = Head(head_size)

        self.time_pooling = time_pooling
       
    def forward(self, x):
        # input: [B x 2 x T x 352], [B x 1 x T x 88]
        # output:
        #   {"head_1": [B x T x 88], 
        #    "head_2": [B x T x 88],...
        # }
        
        if(self.time_pooling):
            src_size = list(x.size())
            src_size[-1] = 88
            x = F.max_pool2d(x, [2,1])

        # => [B x model_size x T x 88]
        y = self.trunk(x)

        output = {}
        for head in self.head_names:
            # => [B x 1 x T x 88]
            output[head] = self.heads[head](y)
            if(self.time_pooling):
                output[head] = F.interpolate(output[head], size=src_size[-2:], mode='bilinear')
            output[head] = torch.clip(output[head], 1e-7, 1-1e-7)

        return output


class HPPNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_size = config['model_size']

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.subnets = {}
        self.subnets['all'] = nn.ModuleList() #[self.subnet_onset, self.subnet_frame]
        if 'onset_subnet' in self.config['SUBNETS_TO_TRAIN']:
            self.subnet_onset = SubNet(model_size, config['onset_subnet_heads'])
            self.subnets['onset_subnet'] = self.subnet_onset
            self.subnets['all'].append(self.subnet_onset)
        if 'frame_subnet' in self.config['SUBNETS_TO_TRAIN']:
            self.subnet_frame = SubNet(model_size, config['frame_subnet_heads'], time_pooling=True)
            self.subnets['frame_subnet'] = self.subnet_frame
            self.subnets['all'].append(self.subnet_frame)
            
        self.inference_mode = False
        self.e = 2**(1/24)
        self.to_cqt = CQT(sr=config["sample_rate"], hop_length=config["hop_length"], fmin=27.5/self.e, n_bins=88*4, \
            bins_per_octave=config["bins_per_semitone"]*12, output_format='Magnitude')
        self.bce = nn.BCELoss()

    
    def forward(self, waveform):
        '''
        inputs:
            waveform [b x num_samples] (sample rate: 16000)
        '''
        y = waveform
        # => [b x T x 352]
        cqt = self.to_cqt(y).permute([0,2, 1]).float()
        cqt_db = self.amplitude_to_db(cqt)
        return self.forward_logspecgram(cqt_db)
    
    def forward_logspecgram(self, cqt_db):
        # inputs: [b x n], [b x T x 88]
        # inputs: cqt_db [b x T x 352]
        # outputs: 
        '''
        {
            "onset":[b x T x 88],
            "frame":
            "offset":
            "velocity":
        }
        '''        
        specgram_db = torch.unsqueeze(cqt_db, dim=1).to(self.config['device'])
        
        if self.inference_mode == False:
            specgram_db = specgram_db[:, :, :self.frame_num, :]
            pad_len = self.frame_num - specgram_db.size()[2]
            if(pad_len > 0):
                print(f'frame len < {self.frame_num}, zero_pad_len:{pad_len}')
                # => [B x 2 x T x 352]
                specgram_db = F.pad(specgram_db, [0, 0, 0, pad_len], mode='replicate')
                assert specgram_db.size()[2] == self.frame_num
                
        # specgram_db = cqt_db
        results = {}
        if 'onset_subnet' in self.config['SUBNETS_TO_TRAIN']:
            results_1 = self.subnet_onset(specgram_db)
            results.update(results_1)
        if 'frame_subnet' in self.config['SUBNETS_TO_TRAIN']:
            results_2 = self.subnet_frame(specgram_db)
            results.update(results_2)
            
        if self.inference_mode == True:
            del results['offset']
        return results
    
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

        # optimizer = torch.optim.Adam([
        #     {"params": self.subnets[], "lr": self.config["lr"]}
        # ])

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

        self.frame_num = y_frame.size()[-2]
        self.piano_roll_size = y_frame.size()[-2:]

        audio_label_reshape = audio.reshape(-1, audio.shape[-1])

        # Forward pass
        results = self.forward(audio_label_reshape)
        predictions = {
            'onset': torch.clip(y_onset, 0, 0),
            'offset': torch.clip(y_offset, 0, 0),
            'frame': torch.clip(y_frame, 0, 0),
            'velocity': torch.clip(y_velocity, 0, 0)
        }

        # Loss
        losses = {}
        if 'onset' in results.keys():
            predictions['onset'] = results['onset'].reshape(*y_onset.shape)
            # [b x T x 88]
            losses['train_onset_loss'] = - 2 * y_onset * torch.log(predictions['onset']) - ( 1 - y_onset) * torch.log(1-predictions['onset'])
            losses['train_onset_loss'] = losses['train_onset_loss'].mean()
        if 'offset' in results.keys():
            predictions['offset'] = results['offset'].reshape(*y_offset.shape)
            losses['train_offset_loss'] = self.bce(predictions['offset'], y_offset)
        if 'frame' in results.keys():
            predictions['frame'] = results['frame'].reshape(*y_frame.shape)
            losses['train_frame_loss'] = self.bce(predictions['frame'] , y_frame).mean()
        if 'velocity' in results.keys():
            predictions['velocity'] = results['velocity'].reshape(*y_velocity.shape)
            losses['train_velocity_loss'] = self.loss_velocity(predictions['velocity'], y_velocity, y_onset)
        
        losses['train_total_loss'] = sum(losses.values())

        self.log_dict({
            losses
        }, logger=True, on_step=False, on_epoch=True)
        return losses['train_total_loss']
    
    def validation_step(self, val_batch, batch_idx):
        audio = val_batch["audio"]
        y_onset = val_batch["onset"]
        y_offset = val_batch["offset"]
        y_velocity = val_batch["velocity"]
        y_frame = val_batch["frame"]
        y_velocity = y_velocity.float()

        self.frame_num = y_frame.size()[-2]
        self.piano_roll_size = y_frame.size()[-2:]

        audio_label_reshape = audio.reshape(-1, audio.shape[-1])

        # Forward pass
        results = self.forward(audio_label_reshape)
        predictions = {
            'onset': torch.clip(y_onset, 0, 0),
            'offset': torch.clip(y_offset, 0, 0),
            'frame': torch.clip(y_frame, 0, 0),
            'velocity': torch.clip(y_velocity, 0, 0)
        }

        # Loss
        losses = {}
        if 'onset' in results.keys():
            predictions['onset'] = results['onset'].reshape(*y_onset.shape)
            # [b x T x 88]
            losses['val_onset_loss'] = - 2 * y_onset * torch.log(predictions['onset']) - ( 1 - y_onset) * torch.log(1-predictions['onset'])
            losses['val_onset_loss'] = losses['train_onset_loss'].mean()
        if 'offset' in results.keys():
            predictions['offset'] = results['offset'].reshape(*y_offset.shape)
            losses['val_offset_loss'] = self.bce(predictions['offset'], y_offset)
        if 'frame' in results.keys():
            predictions['frame'] = results['frame'].reshape(*y_frame.shape)
            losses['val_frame_loss'] = self.bce(predictions['frame'] , y_frame).mean()
        if 'velocity' in results.keys():
            predictions['velocity'] = results['velocity'].reshape(*y_velocity.shape)
            losses['val_velocity_loss'] = self.loss_velocity(predictions['velocity'], y_velocity, y_onset)
        
        losses['val_total_loss'] = sum(losses.values())

        self.log_dict({
            losses
        }, logger=True, on_step=False, on_epoch=True)
        return losses['val_total_loss']
