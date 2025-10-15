"""
    Contains the Trainer functionality for training models 
    supported by the snap2midi package.
"""

from .oaf import train_oaf as oaf_train
from .kong import train_kong as kong_train
from .kong import train_kong_pedals as kong_train_pedals
from .hft import train_hft as hft_train

class Trainer:
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

    def train_oaf(self, batch_size=8, iterations=50000, lr=0.0006, frame_rate=31.25, in_features=229, out_features=88, \
                  learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000, \
                    clip_gradient_norm=3, threshold=0.5, temporal_sizes=[3, 3, 3], freq_sizes=[3, 3, 3], \
                    out_channels=[32, 32, 64], pool_sizes=[1, 2, 2], dropout_probs=[0, 0.25, 0.25], dropout_fc=0.5, \
                    fc_size=512, onset_lstm_units=128, combined_lstm_units=128, resume=0, \
                    checkpoint_name: str = "checkpoint_23.pt", pitch_offset: int = 21):
        """
            Train Onsets and Frames model with specified configuration.

            Args:
                batch_size (int): Batch size for training.
                iterations (int): Number of iterations for training. Default is 50000.
                lr (float): Learning rate for the optimizer. Default is 0.0006.
                frame_rate (float): Frame rate for the model. Default is 31.25.
                in_features (int): Number of input features. Default is 229.
                out_features (int): Number of output features. Default is 88.
                learning_rate_decay_rate (float): Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): Number of steps for learning rate decay. Default is 10000.
                clip_gradient_norm (float): Gradient clipping norm. Default is 3.
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
                resume (int): Whether to resume training from a checkpoint. Default is 0 (False).
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_23.pt".
                pitch_offset (int): Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
        """    
        config = self.build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="OnsetsAndFrames",
            batch_size=batch_size,
            iterations=iterations,
            lr=lr,
            frame_rate=frame_rate,
            in_features=in_features,
            out_features=out_features,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            clip_gradient_norm=clip_gradient_norm,
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
            resume=resume,
            pitch_offset=pitch_offset
        )
        if resume:
            resume_path = f"runs/{checkpoint_name}"
            config["resume_path"] = resume_path

        save_dir = "runs/oaf/"
        config["save_dir"] = save_dir
        oaf_train.main(config)
    
    def train_kong(self, batch_size: int = 4, factors: list = [16, 32, 32], iterations: int = 200000, frame_rate: float = 100, \
                lr: float = 5e-4,  onset_threshold: float = 0.3, offset_threshold: float = 0.3, \
                frame_threshold: float = 0.3, pedal_offset_threshold: float = 0.3, cmp: int = 48, \
                momentum: float = 0.01, learning_rate_decay_rate: float = 0.9, learning_rate_decay_steps: int = 10000, \
                clip_gradient_norm: float = 3.0, resume: int = 0, checkpoint_name: str = "checkpoint_19.pt"):
        """
            Train Kong model with specified configuration.

            Args:
                batch_size (int): Batch size for training. Default is 4.
                factors (list): List of factors for the model. Default is [16, 32, 32].
                iterations (int): Number of iterations for training. Default is 200000.
                frame_rate (float): Frame rate for the model. Default is 100.
                lr (float): Learning rate for the optimizer. Default is 5e-4.
                onset_threshold (float): Threshold for onset detection. Default is 0.3.
                offset_threshold (float): Threshold for offset detection. Default is 0.3.
                frame_threshold (float): Threshold for frame detection. Default is 0.1.
                pedal_offset_threshold (float): Threshold for pedal offset detection. Default is 0.2.
                cmp (int): Contextual margin padding. Default is 48.
                momentum (float): Momentum for the optimizer. Default is 0.01.
                learning_rate_decay_rate (float): Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): Number of steps for learning rate decay. Default is 1000.
                clip_gradient_norm (float): Gradient clipping norm. Default is 3.0.
                resume (int): Whether to resume training from a checkpoint. Default is 0 (False).
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_19.pt".
        """
        config = self.build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="Kong",
            batch_size=batch_size,
            factors=factors,
            iterations=iterations,
            frame_rate=frame_rate,
            lr=lr,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_threshold=frame_threshold,
            pedal_offset_threshold=pedal_offset_threshold,
            cmp=cmp,
            momentum=momentum,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            clip_gradient_norm=clip_gradient_norm,
            resume=resume
        )
        if resume:
            resume_path = f"runs/kong/{checkpoint_name}"
            config["resume_path"] = resume_path

        save_dir = "runs/kong/"
        config["save_dir"] = save_dir
        kong_train.main(config)
    
    def train_kong_pedals(self, batch_size: int = 4, factors: list = [16, 32, 32], iterations: int = 200000, frame_rate: float = 100, \
                lr: float = 5e-4,  onset_threshold: float = 0.3, offset_threshold: float = 0.3, \
                frame_threshold: float = 0.3, pedal_offset_threshold: float = 0.3, cmp: int = 48, \
                momentum: float = 0.01, learning_rate_decay_rate: float = 0.9, learning_rate_decay_steps: int = 10000, \
                clip_gradient_norm: float = 3.0, resume: int = 0, checkpoint_name: str = "checkpoint_19.pt"):
        """
            Train Kong Pedal model with specified configuration.

            Args:
                batch_size (int): Batch size for training. Default is 4.
                factors (list): List of factors for the model. Default is [16, 32, 32].
                iterations (int): Number of iterations for training. Default is 200000.
                frame_rate (float): Frame rate for the model. Default is 100.
                lr (float): Learning rate for the optimizer. Default is 5e-4.
                onset_threshold (float): Threshold for onset detection. Default is 0.3.
                offset_threshold (float): Threshold for offset detection. Default is 0.3.
                frame_threshold (float): Threshold for frame detection. Default is 0.1.
                pedal_offset_threshold (float): Threshold for pedal offset detection. Default is 0.2.
                cmp (int): Contextual margin padding. Default is 48.
                momentum (float): Momentum for the optimizer. Default is 0.01.
                learning_rate_decay_rate (float): Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): Number of steps for learning rate decay. Default is 1000.
                clip_gradient_norm (float): Gradient clipping norm. Default is 3.0.
                resume (int): Whether to resume training from a checkpoint. Default is 0 (False).
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_19.pt".
        """
        config = self.build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="KongPedal",
            batch_size=batch_size,
            factors=factors,
            iterations=iterations,
            frame_rate=frame_rate,
            lr=lr,
            onset_threshold=onset_threshold,
            offset_threshold=offset_threshold,
            frame_threshold=frame_threshold,
            pedal_offset_threshold=pedal_offset_threshold,
            cmp=cmp,
            momentum=momentum,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            clip_gradient_norm=clip_gradient_norm,
            resume=resume
        )
        if resume:
            resume_path = f"runs/kong_pedal/{checkpoint_name}"
            config["resume_path"] = resume_path

        save_dir = "runs/kong_pedal/"
        config["save_dir"] = save_dir
        kong_train_pedals.main(config)

    def train_hft(self, batch_size: int = 4, margin_b: int = 32, margin_f: int = 32, n_bins: int = 256, n_slice: int=16, \
                  num_frame: int = 128, epochs: int = 50, frame_rate: int = 100, num_velocity: int = 128, num_note: int = 88, \
                  lr: float = 1e-4, dropout: float = 0.1, clip_gradient_norm: float = 1.0,seed: int = 1234, \
                  cnn_channel: int = 4, cnn_kernel: int = 5, d: int = 256, pff_dim: int = 512, enc_layer: int = 3, \
                  dec_layer: int = 3, enc_head: int = 4, dec_head: int = 4, weight_A: float = 1.0, weight_B: float = 1.0,\
                  verbose: int = 1, resume: int = 0, checkpoint_name: str = "checkpoint_90.pt"):
        """
            Train hFT-Transformer model with specified configuration.

            Args:
                batch_size (int): Batch size for training. Default is 4.
                margin_b (int): Margin before the input frame. Default is 32.
                margin_f (int): Margin after the input frame. Default is 32.
                n_bins (int): Number of frequency bins in the input feature. Default is 256.
                n_slice (int): Slice dataset into n_slice parts; used for indexing. Default is 16.
                num_frame (int): Number of frames in the input. Default is 128.
                epochs (int): Number of epochs for training. Default is 50.
                frame_rate (int): Frame rate for the model. Default is 100.
                num_velocity (int): Number of velocity levels. Default is 128.
                num_note (int): Number of MIDI notes. Default is 128.
                lr (float): Learning rate for the optimizer. Default is 1e-4.
                dropout (float): Dropout rate for the model. Default is 0.1.
                clip_gradient_norm (float): Gradient clipping norm. Default is 1.0.
                seed (int): Random seed for reproducibility. Default is 1234.
                cnn_channel (int): Number of CNN channels. Default is 4.
                cnn_kernel (int): CNN kernel size. Default is 5.
                d (int): Dimension of the model. Default is 256.
                pff_dim (int): Dimension of the position-wise feed-forward layer. Default is 512.
                enc_layer (int): Number of encoder layers. Default is 3.
                dec_layer (int): Number of decoder layers. Default is 3.
                enc_head (int): Number of attention heads in the encoder. Default is 4.
                dec_head (int): Number of attention heads in the decoder. Default is 4.
                weight_A (float): Weight for loss A. Default is 1.0.
                weight_B (float): Weight for loss B. Default is 1.0.
                verbose (int): Verbosity level. Default is 1.
                resume (int): Whether to resume training from a checkpoint. Default is 0 (False).
                checkpoint_name (str): Name of the checkpoint file to resume from. Default is "checkpoint_90.pt".

            Returns:
                None
        """
        
        config = self.build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="HFT",
            batch_size=batch_size,
            margin_b=margin_b,
            margin_f=margin_f,
            n_bins=n_bins,
            num_note=num_note,
            num_velocity=num_velocity,
            num_frame=num_frame,
            n_slice=n_slice,
            epochs=epochs,
            frame_rate=frame_rate,
            lr=lr,
            dropout=dropout,
            clip_gradient_norm=clip_gradient_norm,
            seed=seed,
            cnn_channel=cnn_channel,
            cnn_kernel=cnn_kernel,
            d=d,
            pff_dim=pff_dim,
            enc_layer=enc_layer,
            dec_layer=dec_layer,
            enc_head=enc_head,
            dec_head=dec_head,
            weight_A=weight_A,
            weight_B=weight_B,
            verbose=verbose,
            resume=resume
        )
        if config["resume"]:
            resume_path = f"runs/{checkpoint_name}"
            config["resume_path"] = resume_path
        
        save_dir = "runs/hft"
        config["save_dir"] = save_dir
        hft_train.main(config)
