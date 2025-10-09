"""
    Contains the Evaluator functionality for evaluating models 
    supported by the snap2midi package.
"""
from .hft.evaluate import evaluate_test as hft_eval
from .oaf.evaluate import evaluate_test as oaf_eval
from .kong.evaluate import evaluate as kong_eval
from .kong.evaluate import evaluate_pedal as kong_pedal_eval

class Evaluator:
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

    def evaluate_oaf(self, frame_rate=31.25, in_features=229, out_features=88, \
                    threshold=0.5, temporal_sizes=[3, 3, 3], freq_sizes=[3, 3, 3], \
                    out_channels=[32, 32, 64], pool_sizes=[1, 2, 2], dropout_probs=[0, 0.25, 0.25], dropout_fc=0.5, \
                    fc_size=512, onset_lstm_units=128, combined_lstm_units=128, \
                    checkpoint_name: str = "checkpoint_23.pt", pitch_offset: int = 21):

        """
            Evaluate Onsets and Frames model with specified configuration.

            Args:
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
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_23.pt".
                pitch_offset (int): Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
        """    
        config = self.build_config_from_kwargs(
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
            pitch_offset=pitch_offset
        )
        checkpoint_path = f"runs/{checkpoint_name}"
        config["checkpoint_path"] = checkpoint_path
        results = oaf_eval(config)
        return results

    def evaluate_kong(self, checkpoint_name: str = "checkpoint_180000.pt", factors: list = [16, 32, 32], \
        frame_rate: float=100, onset_threshold: float = 0.3, offset_threshold: float = 0.3, frame_threshold: float = 0.3, \
        pedal_offset_threshold: float = 0.3, cmp: int=48, momentum: float = 0.01):
        """
            Evaluate Kong model with specified configuration.

            Args:
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_140000.pt".
                factors (list): List of factors for the model. Default is [16, 32, 32].
                frame_rate (float): Frame rate for the model. Default is 100.
                onset_threshold (float): Threshold for onset detection. Default is 0.3.
                offset_threshold (float): Threshold for offset detection. Default is 0.3.
                frame_threshold (float): Threshold for frame detection. Default is 0.3.
                pedal_offset_threshold (float): Threshold for pedal offset detection. Default is 0.3.
                cmp (int): CMP value for the model. Default is 48.
                momentum (float): Momentum value for the model. Default is 0.01.
        """
        config = self.build_config_from_kwargs(
            factors=factors,
            frame_rate=frame_rate,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_threshold=frame_threshold,
            pedal_offset_threshold=pedal_offset_threshold,
            cmp=cmp,
            momentum=momentum
        )
        checkpoint_path = f"runs/kong/{checkpoint_name}"
        config["checkpoint_note_path"] = checkpoint_path
        results = kong_eval(config)
        return results
    
    def evaluate_kong_pedal(self, checkpoint_name: str = "checkpoint_180000.pt", factors: list = [16, 32, 32], \
        frame_rate: float=100, onset_threshold: float = 0.3, offset_threshold: float = 0.3, frame_threshold: float = 0.3, \
        pedal_offset_threshold: float = 0.3, cmp: int=48, momentum: float = 0.01):
        """
            Evaluate Kong Pedal model with specified configuration.

            Args:
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_140000.pt".
                factors (list): List of factors for the model. Default is [16, 32, 32].
                frame_rate (float): Frame rate for the model. Default is 100.
                onset_threshold (float): Threshold for onset detection. Default is 0.3.
                offset_threshold (float): Threshold for offset detection. Default is 0.3.
                frame_threshold (float): Threshold for frame detection. Default is 0.3.
                pedal_offset_threshold (float): Threshold for pedal offset detection. Default is 0.3.
                cmp (int): CMP value for the model. Default is 48.
                momentum (float): Momentum value for the model. Default is 0.01.
        """
        config = self.build_config_from_kwargs(
            factors=factors,
            frame_rate=frame_rate,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_threshold=frame_threshold,
            pedal_offset_threshold=pedal_offset_threshold,
            cmp=cmp,
            momentum=momentum
        )
        checkpoint_path = f"runs/kong_pedal/{checkpoint_name}"
        config["checkpoint_pedal_path"] = checkpoint_path
        results = kong_pedal_eval(config)
        return results

    def evaluate_hft(self, checkpoint_name: str = "checkpoint_7.pt", margin_b: int = 32, margin_f: int = 32, \
                  n_bins: int = 256, n_slice: int=16, frame_threshold: float = 0.5, onset_threshold: float = 0.5, \
                  offset_threshold: float = 0.5, num_frame: int = 128, epochs: int = 50, frame_rate: int = 100, \
                  num_velocity: int = 128, note_min: int = 21, note_max: int = 108, hop_sample: int = 256, sr: int = 16000, 
                  cnn_channel: int = 4, cnn_kernel: int = 5, d: int = 256, pff_dim: int = 512, 
                  enc_layer: int = 3, dropout: float = 0.1, dec_layer: int = 3, enc_head: int = 4, dec_head: int = 4, \
                  weight_A: float = 1.0, weight_B: float = 1.0):
        """
            Evaluate HFT model with specified configuration.

            Args:
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_7.pt".
                margin_b (int): Back margin for input feature. Default is 32.
                margin_f (int): Front margin for input feature. Default is 32.
                n_bins (int): Number of frequency bins for input feature. Default is 256.
                n_slice (int): Number of slices for input feature. Default is 16.
                frame_threshold (float): Threshold for frame prediction. Default is 0.5.
                onset_threshold (float): Threshold for onset prediction. Default is 0.5.
                offset_threshold (float): Threshold for offset prediction. Default is 0.5.
                num_frame (int): Number of frames for input feature. Default is 128.
                epochs (int): Number of training epochs. Default is 50.
                frame_rate (int): Frame rate for the model. Default is 100.
                num_velocity (int): Number of velocity levels. Default is 128.
                note_min (int): Minimum MIDI note number. Default is 21.
                note_max (int): Maximum MIDI note number. Default is 108.
                hop_sample (int): Hop size in samples for input feature. Default is 256.
                sr (int): Sampling rate for input feature. Default is 16000.
                cnn_channel (int): Number of channels for CNN layers. Default is 4.
                cnn_kernel (int): Kernel size for CNN layers. Default is 5.
                d (int): Dimension of model. Default is 256.
                pff_dim (int): Dimension of position-wise feedforward layers. Default is 512.
                enc_layer (int): Number of encoder layers. Default is 3.
                dropout (float): Dropout probability. Default is 0.1.
                dec_layer (int): Number of decoder layers. Default is 3.
                enc_head (int): Number of attention heads in encoder. Default is 4.
                dec_head (int): Number of attention heads in decoder. Default is 4.
                weight_A (float): Weight for loss A. Default is 1.0.
                weight_B (float): Weight for loss B. Default is 1.0.
        """
        config = self.build_config_from_kwargs( 
            margin_b=margin_b,
            margin_f=margin_f,
            n_bins=n_bins,
            n_slice=n_slice,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            num_frame=num_frame,
            epochs=epochs,
            frame_rate=frame_rate,
            num_velocity=num_velocity,
            note_min=note_min,
            note_max=note_max,
            num_note=note_max - note_min + 1,
            hop_sample=hop_sample,
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
            weight_B=weight_B
        )
        checkpoint_path = f"runs/{checkpoint_name}"
        config["checkpoint_path"] = checkpoint_path
        results = hft_eval(config)
        return results
