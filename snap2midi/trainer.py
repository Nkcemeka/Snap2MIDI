"""
    Contains the Trainer functionality for training models 
    supported by the snap2midi package.
"""

from .models.oaf import train_oaf as oaf_train
from .models.oafv2 import train_oafv2 as oafv2_train
from .models.kong import train_kong as kong_train
from .models.kong import train_kong_pedals as kong_train_pedals
from .models.hft import train_hft as hft_train
from .models.transkun import train_transkun as transkun_train
from .models.hpp import train_hpp as hpp_train

class Trainer:
    """
        Trainer class to handle training of different models.
        Models supported: Onsets and Frames (OAF), Kong, Kong Pedals, hFT-Transformer.
    """
    def __init__(self):
        pass

    def _build_config_from_kwargs(self, **kwargs):
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

    def train_oaf(self, base_path: str="./data/oaf/", batch_size=8, iterations=50000, lr=0.0006, \
                frame_rate=31.25, in_features=229, out_features=88, learning_rate_decay_rate=0.98,\
                learning_rate_decay_steps=10000, clip_gradient_norm=3, threshold=0.5, \
                temporal_sizes=[3, 3, 3], freq_sizes=[3, 3, 3], out_channels=[32, 32, 64], \
                pool_sizes=[1, 2, 2], dropout_probs=[0, 0.25, 0.25], dropout_fc=0.5, \
                fc_size=512, onset_lstm_units=128, combined_lstm_units=128, pitch_offset: int = 21, \
                num_workers: int=4, num_nodes: int=1, logger_name: str='csv', resume_path:str|None=None, \
                save_dir: str="./save_dir"):
        """
            Train Onsets and Frames model with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted training data
                batch_size (int): 
                    Batch size for training/validation.
                iterations (int): 
                    Number of iterations for training. Default is 50000.
                lr (float): 
                    Learning rate for the optimizer. Default is 0.0006.
                frame_rate (float): 
                    Frame rate for the model. Default is 31.25.
                in_features (int): 
                    Number of input features. Default is 229.
                out_features (int):
                    Number of output features. Default is 88.
                learning_rate_decay_rate (float): 
                    Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): 
                    Number of steps for learning rate decay. Default is 10000.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 3.
                threshold (float): 
                    Threshold for onset detection. Default is 0.5.
                temporal_sizes (list): 
                    List of temporal sizes for convolutional layers. Default is [3, 3, 3].
                freq_sizes (list): 
                    List of frequency sizes for convolutional layers. Default is [3, 3, 3].
                out_channels (list): 
                    List of output channels for convolutional layers. Default is [32, 32, 64].
                pool_sizes (list): 
                    List of pooling sizes for convolutional layers. Default is [1, 2, 2].
                dropout_probs (list): 
                    List of dropout probabilities for convolutional layers. Default is [0, 0.25, 0.25].
                dropout_fc (float): 
                    Dropout probability for fully connected layer. Default is 0.5.
                fc_size (int): 
                    Size of fully connected layer. Default is 512.
                onset_lstm_units (int): 
                    Number of LSTM units for onset detection. Default is 128.
                combined_lstm_units (int): 
                    Number of LSTM units for combined model. Default is 128.
                pitch_offset (int): 
                    Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
                num_workers (int):
                    Number of workers. Defualt is 4.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.
                
            Returns
            --------
                None
        """    
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="OnsetsAndFrames",
            base_path=base_path,
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
            pitch_offset=pitch_offset,
            num_workers=num_workers,
            num_nodes=num_nodes,
            logger_name=logger_name,
            resume_path=resume_path,
            save_dir=save_dir
        )
        oaf_train.main(config)
    
    def train_oafv2(self, base_path: str="./data/oafv2/", batch_size=8, iterations=500000, lr=0.0006, \
        sequence_length: int=327680, seed: int=42, sample_rate: int=16000, n_fft: int=2048, \
        n_mels: int=229, htk: bool=True, fmin: int=32, \
        hop_length: int=512, fmax: int|None=None, pad_mode: str="reflect", center: bool=True, \
        window: str="hann", in_features=229, out_features=88, model_complexity: int=48,\
        learning_rate_decay_rate=0.98, learning_rate_decay_steps=10000, clip_gradient_norm=3, \
        pitch_offset: int = 21, num_workers: int=4, num_nodes: int=1, \
        logger_name: str='csv', resume_path:str|None=None, \
        save_dir: str="./save_dir"):
        """
            Train Onsets and Frames model version 2 with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted training data
                batch_size (int): 
                    Batch size for training/validation. Default is 8.
                iterations (int): 
                    Number of iterations for training. Default is 500000.
                lr (float): 
                    Learning rate for the optimizer. Default is 0.0006.
                sample_rate (int):
                    Sample rate. Default is 16000.
                n_fft (int):
                    Size of fft window.
                n_mels (int):
                    Number of mel bands.
                htk (bool):
                    Use htk for mel spectrogram.
                fmin (int):
                    Min. frequeny for FFT
                fmax (int | None):
                    Max frequency for FFT.
                pad_mode (str):
                    Pad mode for FFT. Default is reflect.
                center (str):
                    Center window for FFT computation
                window (str):
                    Window for FFT. Default is 'hann'.
                seed (int):
                    Seed for sampling from dataset during training.
                in_features (int): 
                    Number of input features. Default is 229.
                out_features (int):
                    Number of output features. Default is 88.
                model_complexity (int):
                    Model complexity. Default is 48.
                learning_rate_decay_rate (float): 
                    Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): 
                    Number of steps for learning rate decay. Default is 10000.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 3.
                pitch_offset (int): 
                    Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
                num_workers (int):
                    Number of workers. Defualt is 4.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.
                
            Returns
            --------
                None
        """    
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="OnsetsAndFramesV2",
            base_path=base_path,
            batch_size=batch_size,
            iterations=iterations,
            lr=lr,
            sequence_length=sequence_length,
            seed=seed,
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            htk=htk,
            fmin=fmin,
            hop_length=hop_length,
            fmax=fmax,
            pad_mode=pad_mode,
            center=center,
            window=window,
            in_features=in_features,
            out_features=out_features,
            model_complexity=model_complexity,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            clip_gradient_norm=clip_gradient_norm,
            pitch_offset=pitch_offset,
            num_workers=num_workers,
            num_nodes=num_nodes,
            logger_name=logger_name,
            resume_path=resume_path,
            save_dir=save_dir
        )
        oafv2_train.main(config)

    def train_kong(self, base_path: str="./data/kong/", batch_size: int = 4, factors: list = [16, 32, 32], iterations: int = 200000, frame_rate: float = 100, \
                lr: float = 5e-4,  onset_threshold: float = 0.3, offset_threshold: float = 0.3, \
                frame_threshold: float = 0.3, pedal_offset_threshold: float = 0.3, cmp: int = 48, \
                momentum: float = 0.01, learning_rate_decay_rate: float = 0.9, learning_rate_decay_steps: int = 10000, \
                clip_gradient_norm: float = 3.0, num_workers: int=4, logger_name: str='csv',\
                val_steps: int=5000, num_nodes: int=1, \
                resume_path:str|None=None, save_dir: str="./save_dir"):
        """
            Train Kong model with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted data
                batch_size (int): 
                    Batch size for training. Default is 4.
                factors (list): 
                    List of factors for the model. Default is [16, 32, 32].
                iterations (int): 
                    Number of iterations for training. Default is 200000.
                frame_rate (float): 
                    Frame rate for the model. Default is 100.
                lr (float): 
                    Learning rate for the optimizer. Default is 5e-4.
                onset_threshold (float): 
                    Threshold for onset detection. Default is 0.3.
                offset_threshold (float): 
                    Threshold for offset detection. Default is 0.3.
                frame_threshold (float): 
                    Threshold for frame detection. Default is 0.1.
                pedal_offset_threshold (float): 
                    Threshold for pedal offset detection. Default is 0.2.
                cmp (int): 
                    Contextual margin padding. Default is 48.
                momentum (float): 
                    Momentum for the optimizer. Default is 0.01.
                learning_rate_decay_rate (float): 
                    Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): 
                    Number of steps for learning rate decay. Default is 1000.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 3.0.
                num_workers (int):
                    Number of workers. Defualt is 4.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                val_steps (int):
                    How many N steps before performing validation.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.
            
            Returns
            ---------
                None
        """
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="Kong",
            base_path=base_path,
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
            resume_path=resume_path,
            num_workers=num_workers,
            val_steps=val_steps,
            num_nodes=num_nodes,
            logger_name=logger_name,
            save_dir=save_dir,
        )
        kong_train.main(config)
    
    def train_kong_pedals(self, base_path: str="./data/kong_pedal/", batch_size: int = 4, factors: list = [16, 32, 32], iterations: int = 200000, frame_rate: float = 100, \
                lr: float = 5e-4,  onset_threshold: float = 0.3, offset_threshold: float = 0.3, \
                frame_threshold: float = 0.3, pedal_offset_threshold: float = 0.3, cmp: int = 48, \
                momentum: float = 0.01, learning_rate_decay_rate: float = 0.9, learning_rate_decay_steps: int = 10000, \
                clip_gradient_norm: float = 3.0, num_workers: int=4, logger_name: str='csv', \
                val_steps: int=5000, num_nodes: int=1, \
                resume_path:str|None=None, save_dir: str="./save_dir"):
        """
            Train Kong Pedal model with specified configuration.

            Parameters
            ----------
                batch_size (int): 
                    Batch size for training. Default is 4.
                factors (list): 
                    List of factors for the model. Default is [16, 32, 32].
                iterations (int): 
                    Number of iterations for training. Default is 200000.
                frame_rate (float): 
                    Frame rate for the model. Default is 100.
                lr (float): 
                    Learning rate for the optimizer. Default is 5e-4.
                onset_threshold (float): 
                    Threshold for onset detection. Default is 0.3.
                offset_threshold (float): 
                    Threshold for offset detection. Default is 0.3.
                frame_threshold (float): 
                    Threshold for frame detection. Default is 0.1.
                pedal_offset_threshold (float): 
                    Threshold for pedal offset detection. Default is 0.2.
                cmp (int): 
                    Contextual margin padding. Default is 48.
                momentum (float): 
                    Momentum for the optimizer. Default is 0.01.
                learning_rate_decay_rate (float): 
                    Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): 
                    Number of steps for learning rate decay. Default is 1000.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 3.0.
                num_workers (int):
                    Number of workers. Defualt is 4.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                val_steps (int):
                    How many N steps before performing validation.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.

            Returns
            ---------
                None
        """
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="KongPedal",
            base_path=base_path,
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
            resume_path=resume_path,
            num_nodes=num_nodes,
            val_steps=val_steps,
            save_dir=save_dir,
            num_workers=num_workers,
            logger_name=logger_name
        )
        kong_train_pedals.main(config)

    def train_hft(self, base_path: str="./data/hft/", batch_size: int = 4, n_div_train: int=1, n_div_val: int=1, margin_b: int = 32, margin_f: int = 32, n_bins: int = 256, n_slice: int=16, \
        num_frame: int = 128, epochs: int = 50, frame_rate: int = 100, num_velocity: int = 128, num_note: int = 88, \
        lr: float = 1e-4, dropout: float = 0.1, clip_gradient_norm: float = 1.0,seed: int = 1234, \
        cnn_channel: int = 4, cnn_kernel: int = 5, d: int = 256, pff_dim: int = 512, enc_layer: int = 3, \
        dec_layer: int = 3, enc_head: int = 4, dec_head: int = 4, weight_A: float = 1.0, weight_B: float = 1.0,\
        verbose: int = 1, num_workers: int=4, logger_name: str='csv', num_nodes: int=1, \
        resume_path:str|None=None, save_dir: str="./save_dir"):
        """
            Train hFT-Transformer model with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted data
                batch_size (int): 
                    Batch size for training. Default is 4.
                n_div_train (int):
                    Number of training divisions
                n_div_val (int):
                    Number of validation divisions
                margin_b (int): 
                    Margin before the input frame. Default is 32.
                margin_f (int): 
                    Margin after the input frame. Default is 32.
                n_bins (int): 
                    Number of frequency bins in the input feature. Default is 256.
                n_slice (int): 
                    Slice dataset into n_slice parts; used for indexing. Default is 16.
                num_frame (int): 
                    Number of frames in the input. Default is 128.
                epochs (int): 
                    Number of epochs for training. Default is 50.
                frame_rate (int): 
                    Frame rate for the model. Default is 100.
                num_velocity (int): 
                    Number of velocity levels. Default is 128.
                num_note (int): 
                    Number of MIDI notes. Default is 128.
                lr (float): 
                    Learning rate for the optimizer. Default is 1e-4.
                dropout (float): 
                    Dropout rate for the model. Default is 0.1.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 1.0.
                seed (int): 
                    Random seed for reproducibility. Default is 1234.
                cnn_channel (int): 
                    Number of CNN channels. Default is 4.
                cnn_kernel (int): 
                    CNN kernel size. Default is 5.
                d (int): 
                    Dimension of the model. Default is 256.
                pff_dim (int): 
                    Dimension of the position-wise feed-forward layer. Default is 512.
                enc_layer (int): 
                    Number of encoder layers. Default is 3.
                dec_layer (int): 
                    Number of decoder layers. Default is 3.
                enc_head (int): 
                    Number of attention heads in the encoder. Default is 4.
                dec_head (int): 
                    Number of attention heads in the decoder. Default is 4.
                weight_A (float): 
                    Weight for loss A. Default is 1.0.
                weight_B (float): 
                    Weight for loss B. Default is 1.0.
                verbose (int): 
                    Verbosity level. Default is 1.
                num_workers (int):
                    Number of workers. Defualt is 4.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.

            Returns
            ---------
                None
        """
        
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="HFT",
            base_path=base_path,
            batch_size=batch_size,
            n_div_train=n_div_train,
            n_div_val=n_div_val,
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
            num_workers=num_workers,
            logger_name=logger_name,
            num_nodes=num_nodes,
            resume_path=resume_path,
            save_dir=save_dir
        )
        hft_train.main(config)

    def train_transkun(self, base_path: str="./data/transkun/", batch_size: int = 4, epochs: int = 1000000,\
        sample_rate: float = 44100, num_workers: int=4, logger_name: str='csv', num_nodes: int=1, 
        val_steps: int=495, freq: int=3000, nProcess: int=1, resume_path:str|None=None, save_dir: str="./save_dir"):
        """
            Train the Transkun model.

            Parameters
            ----------
                base_path (str):
                    Path to extracted data
                batch_size (int): 
                    Batch size for training. Default is 4.
                epochs (int): 
                    Number of epochs for training. Default is 10.
                sample_rate (float):
                    Sample rate of the dataset.
                num_workers (int):
                    Number of workers. Defualt is 4.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                val_steps (int):
                    How many N steps before performing validation.
                freq (int):
                    Frequency at which to compute stats for training.
                nProcess (int):
                    Number of processes. Default is 1.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.

            Returns
            ---------
                None
        """
        import time
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name="Transkun",
            base_path=base_path,
            batch_size=batch_size,
            epochs=epochs,
            seed=time.time(),
            sample_rate=sample_rate,
            freq=freq,
            num_workers=num_workers,
            logger_name=logger_name,
            num_nodes=num_nodes,
            resume_path=resume_path,
            val_steps=val_steps,
            nProcess=nProcess,
            save_dir=save_dir
        )
        transkun_train.main(config)
    
    def train_hpp(self, base_path: str="./data/hpp/", model_type: str="sp", batch_size=4, lr=0.0006, iterations: int=600000,\
        sequence_length: int=327680, seed: int=42, sample_rate: int=16000, \
        bins_per_semitone: int = 4, hop_length: int=320, learning_rate_decay_rate=0.98, \
        learning_rate_decay_steps=10000, clip_gradient_norm=3, \
        pitch_offset: int = 21, num_workers: int=4, num_nodes: int=1, \
        logger_name: str='csv', resume_path:str|None=None, \
        save_dir: str="./save_dir"):
        """
            Train HPP with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted training data
                model_type (str):
                    Supported types are 'sp', 'base', 'tiny', 'ultra-tiny'.
                batch_size (int): 
                    Batch size for training/validation. Default is 8.
                iterations (int): 
                    Number of iterations for training. Default is 500000.
                lr (float): 
                    Learning rate for the optimizer. Default is 0.0006.
                sample_rate (int):
                    Sample rate. Default is 16000.
                bins_per_semitone (int):
                    Bins per semitone for CQT computation.
                seed (int):
                    Seed for sampling from dataset during training.
                learning_rate_decay_rate (float): 
                    Learning rate decay rate. Default is 0.98.
                learning_rate_decay_steps (int): 
                    Number of steps for learning rate decay. Default is 10000.
                clip_gradient_norm (float): 
                    Gradient clipping norm. Default is 3.
                pitch_offset (int): 
                    Pitch offset for MIDI notes. Default is 21. Used to evalutate test set.
                num_workers (int):
                    Number of workers. Defualt is 4.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                logger_name (str):
                    Logger to use in pytorch_lightning. Default is `csv`
                resume_path (str | None): 
                    Whether to resume training from a checkpoint. Default is None. If 
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.
                
            Returns
            --------
                None
        """    
        config = self._build_config_from_kwargs(
            project_name="snap2midi",
            experiment_name=f"HPPNet_{model_type}",
            base_path=base_path,
            batch_size=batch_size,
            iterations=iterations,
            lr=lr,
            sequence_length=sequence_length,
            seed=seed,
            sample_rate=sample_rate,
            bins_per_semitone=bins_per_semitone,
            hop_length=hop_length,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            clip_gradient_norm=clip_gradient_norm,
            pitch_offset=pitch_offset,
            num_workers=num_workers,
            num_nodes=num_nodes,
            logger_name=logger_name,
            resume_path=resume_path,
            save_dir=save_dir
        )

        if model_type == "sp":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet', 'frame_subnet']
            config["onset_subnet_heads"] = ['onset']
            config["frame_subnet_heads"]= ['frame', 'offset', 'velocity']
            config["fixed_dilation"] = 24
            config["model_size"] = 128
        elif model_type == "base":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 128
            config["iterations"] = 600000
        elif model_type == "tiny":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 64
            config["iterations"] = 600000
        elif model_type == "ultra-tiny":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 48
            config["iterations"] = 600000
        else:
            raise RuntimeError(f"Mode type: {model_type} not supported!")
        hpp_train.main(config)
