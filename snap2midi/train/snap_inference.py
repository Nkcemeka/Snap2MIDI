"""
    For performing inference on models supported by the snap2midi package.
"""
from .oaf.inference import inference as oaf_infer
from .hft.inference import inference as hft_infer
from .kong.inference import inference as kong_infer

class Inference:
    def __init__(self):
        pass

    def build_config_from_kwargs(self, **kwargs):
        """
            Build configuration dictionary from keyword arguments.

            Args:
                **kwargs: Keyword arguments to build the configuration dictionary.

            Returns:
                config (dict): Configuration dictionary.
        """
        config = {}
        for key, value in kwargs.items():
            config[key] = value
        return config

    def inference_oaf(self, audio_path, checkpoint_path: str, filename: str = "output",  sample_rate: int = 16000,\
                    frame_rate=31.25, in_features=229, out_features=88, threshold=0.5, temporal_sizes=[3, 3, 3], \
                    freq_sizes=[3, 3, 3], out_channels=[32, 32, 64], pool_sizes=[1, 2, 2], dropout_probs=[0, 0.25, 0.25],\
                    dropout_fc=0.5, fc_size=512, onset_lstm_units=128, combined_lstm_units=128, pitch_offset: int = 21, \
                    mel_n_fft: int = 2048, hop_length: int = 512, window_size: float = 20.0, feature_str: str = "mel"):

        """
            Perform inference using Onsets and Frames model with specified configuration.

            Args:
                audio_path (str): Path to the input audio file.
                filename (str): Name of the output MIDI file. Default is "output.mid".
                checkpoint_path (str): Path to the model checkpoint file. Default is "runs/checkpoint_90.pt".
                sample_rate (int): Sample rate of the input audio. Default is 16000.
                frame_rate (float): Frame rate for the model. Default is 31.25.
                in_features (int): Number of input features. Default is 229.
                out_features (int): Number of output features. Default is 88.
                threshold (float): Threshold for onset detection. Default is 0.5.
                temporal_sizes (list): List of temporal sizes for convolutional layers. Default is [3, 3, 3].
                freq_sizes (list): List of frequency sizes for convolutional layers. Default is [3, 3, 3].
                out_channels (list): List of output channels for convolutional layers. Default is [32, 32, 64].
                pool_sizes (list): List of pooling sizes for convolutional layers. Default is [1, 2, 2].
                dropout_probs (list): List of dropout probabilities for convolutional layers. Default is [0, 0.25, 0.25].
                dropout_fc (float): Dropout probability for fully connected layer. Default is 0.5.
                fc_size (int): Size of fully connected layer. Default is 512.
                onset_lstm_units (int): Number of LSTM units for onset detection. Default is 128.
                combined_lstm_units (int): Number of LSTM units for combined model. Default is 128.
                pitch_offset (int): Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
                mel_n_fft (int): Number of FFT components for Mel spectrogram. Default is 2048.
                hop_length (int): Hop length for Mel spectrogram. Default is 512.
                window_size (float): Window size for feature extraction in seconds. Default is 20.0.
                feature_str (str): Type of feature to extract. Default is "mel".
        """
        config = self.build_config_from_kwargs(
            audio_path=audio_path,
            filename=filename,
            checkpoint_path=checkpoint_path,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            in_features=in_features,
            out_features=out_features,
            threshold=threshold,
            temporal_sizes=temporal_sizes,
            freq_sizes=freq_sizes,
            out_channels=out_channels,
            pool_sizes=pool_sizes,
            dropout_probs=dropout_probs,
            dropout_fc=dropout_fc,
            fc_size=fc_size,
            onset_lstm_units=onset_lstm_units,
            combined_lstm_units=combined_lstm_units,
            pitch_offset=pitch_offset,
            mel_n_fft=mel_n_fft,
            hop_length=hop_length,
            window_size=window_size,
            feature_str=feature_str
        )
        return oaf_infer(config)

    def inference_kong(self, audio_path: str, checkpoint_note_path: str, checkpoint_pedal_path: str, filename: str|None = "output", factors: list = [16, 32, 32], \
        frame_rate: float=100, onset_threshold: float = 0.3, offset_threshold: float = 0.3, frame_threshold: float = 0.3, \
        pedal_offset_threshold: float = 0.3, cmp: int=48, momentum: float = 0.01, sample_rate: int = 16000, \
        window_size: float = 10.0, min_pitch: int = 21, user_ext_config: dict|None=None):
        """
            Perform inference using Kong's model with specified configuration.

            Args:
                audio_path (str): Path to the input audio file.
                checkpoint_note_path (str): Path to the note model checkpoint file.
                checkpoint_pedal_path (str): Path to the pedal model checkpoint file.
                filename (str): Name of the output MIDI file without extension. Default is "output".
                factors (list): List of factors for the model. Default is [16, 32, 32].
                frame_rate (float): Frame rate for the model. Default is 100.
                onset_threshold (float): Threshold for onset detection. Default is 0.3.
                offset_threshold (float): Threshold for offset detection. Default is 0.3.
                frame_threshold (float): Threshold for frame detection. Default is 0.3.
                pedal_offset_threshold (float): Threshold for pedal offset detection. Default is 0.3.
                cmp (int): Number of components for PCA. Default is 48.
                momentum (float): Momentum for batch normalization. Default is 0.01.
                sample_rate (int): Sample rate of the input audio. Default is 16000.
                window_size (float): Window size for feature extraction in seconds. Default is 10.0.
                min_pitch (int): Minimum MIDI pitch number. Default is 21.
                user_ext_config (dict|None): User provided extraction configuration. If None, default config is used.
                                             Default params are {'n_mels': 229, 'max_pitch': 88, 'min_pitch': 21, 
                                             'sample_rate': 16000, 'frame_rate': 100, 'mel_n_fft': 2048}
        """
        config = self.build_config_from_kwargs(
            audio_path=audio_path,
            checkpoint_note_path=checkpoint_note_path,
            checkpoint_pedal_path=checkpoint_pedal_path,
            filename=filename,
            factors=factors,
            frame_rate=frame_rate,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_threshold=frame_threshold,
            pedal_offset_threshold=pedal_offset_threshold,
            cmp=cmp,
            momentum=momentum,
            sample_rate=sample_rate,
            window_size=window_size,
            min_pitch=min_pitch,
            user_ext_config=user_ext_config
        )
        return kong_infer(config)

    def inference_hft(self, audio_path: str, checkpoint_path: str, filename: str = "output", margin_b: int = 32, margin_f: int = 32, \
                  n_bins: int = 256, fft_bins: int = 2048, window_length: int = 2048, n_slice: int=16, frame_threshold: float = 0.5, \
                  onset_threshold: float = 0.5, offset_threshold: float = 0.5, num_frame: int = 128, epochs: int = 50, \
                  frame_rate: int = 100, shift: int = 32, num_velocity: int = 128, note_min: int = 21, note_max: int = 108, \
                  hop_sample: int = 256, mel_bins: int = 256, sr: int = 16000, cnn_channel: int = 4, cnn_kernel: int = 5, d: int = 256, \
                  pff_dim: int = 512, enc_layer: int = 3, dropout: float = 0.1, dec_layer: int = 3, enc_head: int = 4, \
                  dec_head: int = 4, weight_A: float = 1.0, weight_B: float = 1.0, log_offset: float = 1e-8, \
                  pad_mode: str = "constant"):
        """
            Perform inference using HFT model with specified configuration.
            Args:
                audio_path (str): Path to the input audio file.
                checkpoint_path (str): Path to the model checkpoint file.
                filename (str): Name of the output MIDI file without extension. Default is "output".
                margin_b (int): Back margin for input feature. Default is 32.
                margin_f (int): Front margin for input feature. Default is 32.
                n_bins (int): Number of frequency bins for input feature. Default is 256.
                fft_bins (int): Number of FFT components for Mel spectrogram. Default is 2048.
                window_length (int): Window length for Mel spectrogram. Default is 2048.
                n_slice (int): Number of slices for input feature. Default is 16.
                frame_threshold (float): Threshold for frame prediction. Default is 0.5.
                onset_threshold (float): Threshold for onset prediction. Default is 0.5.
                offset_threshold (float): Threshold for offset prediction. Default is 0.5.
                num_frame (int): Number of frames for input feature. Default is 128.
                epochs (int): Number of training epochs. Default is 50.
                frame_rate (int): Frame rate for the model. Default is 100.
                shift (int): Shift or offset for HFT inference. Default is 32.
                num_velocity (int): Number of velocity levels. Default is 128.
                note_min (int): Minimum MIDI note number. Default is 21.
                note_max (int): Maximum MIDI note number. Default is 108.
                hop_sample (int): Hop length for Mel spectrogram. Default is 256.
                mel_bins (int): Number of Mel frequency bins. Default is 256.
                sr (int): Sample rate of the input audio. Default is 16000.
                cnn_channel (int): Number of CNN channels. Default is 4.
                cnn_kernel (int): CNN kernel size. Default is 5.
                d (int): Model dimension. Default is 256.
                pff_dim (int): Position-wise feed-forward dimension. Default is 512.
                enc_layer (int): Number of encoder layers. Default is 3.
                dropout (float): Dropout probability. Default is 0.1.
                dec_layer (int): Number of decoder layers. Default is 3.
                enc_head (int): Number of encoder attention heads. Default is 4.
                dec_head (int): Number of decoder attention heads. Default is 4.
                weight_A (float): Weight for loss A. Default is 1.0.
                weight_B (float): Weight for loss B. Default is 1.0.
                log_offset (float): Log offset for feature extraction. Default is 1e-8.
                pad_mode (str): Padding mode for input feature. Default is "constant".
        """
        config = self.build_config_from_kwargs(
            audio_path=audio_path,
            checkpoint_path=checkpoint_path,
            filename=filename,
            margin_b=margin_b,
            margin_f=margin_f,
            n_bins=n_bins,
            fft_bins=fft_bins,
            window_length=window_length,
            n_slice=n_slice,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            num_frame=num_frame,
            epochs=epochs,
            frame_rate=frame_rate,
            shift=shift,
            num_velocity=num_velocity,
            note_min=note_min,
            note_max=note_max,
            num_note=note_max - note_min + 1,
            hop_sample=hop_sample,
            mel_bins=mel_bins,
            sr=sr,
            cnn_channel=cnn_channel,
            cnn_kernel=cnn_kernel,
            d=d,
            pff_dim=pff_dim,
            enc_layer=enc_layer,
            dropout=dropout,
            dec_layer=dec_layer,
            enc_head=enc_head,
            dec_head=dec_head,
            weight_A=weight_A,
            weight_B=weight_B,
            log_offset=log_offset,
            pad_mode=pad_mode
        )
        return hft_infer(config)
