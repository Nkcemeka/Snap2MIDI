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
                save_dir: str="./save_dir", feature: str="mel", sample_rate: int=16000, \
                max_frame_secs: float=20.0, n_mels: int=229, mel_n_fft: int=2048, \
                hop_length: int=512, seed: int=1234, augment: bool=False, \
                augment_asset_root: str|None=None, augment_manifest_dir: str|None=None, \
                reverb_level: str="rms"):
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
                feature (str), sample_rate (int), max_frame_secs (float),
                n_mels (int), mel_n_fft (int), hop_length (int):
                    The feature parameters this store was extracted with.
                    OAF trains on the feature written at extraction time, so
                    augmentation has to rebuild it from the augmented waveform;
                    these have to match the extract_oaf call that wrote the
                    store, and the dataset verifies that they do. Defaults
                    match extract_oaf's own defaults.
                seed (int):
                    Salts the per-excerpt augmentation seed. Default is 1234.
                augment (bool):
                    Apply Edwards et al.'s augmentation to training excerpts on
                    the fly. Validation is never augmented. Default is False.
                augment_asset_root (str | None):
                    Directory holding room_ir/ and bg_noise/. Required when
                    augment is True.
                augment_manifest_dir (str | None):
                    Overrides the packaged asset manifests. Default is None.
                reverb_level (str):
                    How the reverb stage sets its output level. "rms" preserves
                    the excerpt's energy, as Kaldi's wav-reverberate does,
                    removing the loudness shortcut; "peak" reproduces
                    audiomentations and therefore Edwards. Default is "rms".

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
            save_dir=save_dir,
            feature=feature,
            sample_rate=sample_rate,
            max_frame_secs=max_frame_secs,
            n_mels=n_mels,
            mel_n_fft=mel_n_fft,
            hop_length=hop_length,
            seed=seed,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            reverb_level=reverb_level
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
        save_dir: str="./save_dir", augment: bool=False, augment_asset_root: str|None=None, \
        augment_manifest_dir: str|None=None, reverb_level: str="rms"):
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
                augment (bool):
                    Apply Edwards et al.'s augmentation to training excerpts on
                    the fly. Validation is never augmented. Default is False.
                augment_asset_root (str | None):
                    Directory holding room_ir/ and bg_noise/. Required when
                    augment is True.
                augment_manifest_dir (str | None):
                    Overrides the packaged asset manifests. Default is None.
                reverb_level (str):
                    How the reverb stage sets its output level. "rms" preserves
                    the excerpt's energy, as Kaldi's wav-reverberate does,
                    removing the loudness shortcut; "peak" reproduces
                    audiomentations and therefore Edwards. Default is "rms".

                
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
            save_dir=save_dir,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            reverb_level=reverb_level
        )
        oafv2_train.main(config)

    def train_kong(self, base_path: str="./data/kong/", batch_size: int = 4, factors: list = [16, 32, 32], iterations: int = 200000, frame_rate: float = 100, \
                lr: float = 5e-4,  onset_threshold: float = 0.3, offset_threshold: float = 0.3, \
                frame_threshold: float = 0.3, pedal_offset_threshold: float = 0.3, cmp: int = 48, \
                momentum: float = 0.01, learning_rate_decay_rate: float = 0.9, learning_rate_decay_steps: int = 10000, \
                clip_gradient_norm: float = 3.0, num_workers: int=4, logger_name: str='csv',\
                val_steps: int=5000, num_nodes: int=1, \
                resume_path:str|None=None, save_dir: str="./save_dir", seed: int=1234, \
                augment: bool=False, augment_asset_root: str|None=None, \
                augment_manifest_dir: str|None=None, reverb_level: str="rms", \
                experiment_name: str="Kong", log_steps: int|None=None):
        """
            Train Kong model with specified configuration.

            Augmentation is off by default. Turning it on applies Edwards et
            al.'s pipeline to the training excerpts only; validation stays
            clean, since checkpoint selection monitors valid_total_loss.
            reverb_level selects how the reverb stage sets its output level:
            "rms", the default, preserves the excerpt's energy as Kaldi's
            wav-reverberate does; "peak" reproduces audiomentations and
            therefore Edwards.

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
            experiment_name=experiment_name,
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
            log_steps=log_steps,
            num_nodes=num_nodes,
            logger_name=logger_name,
            save_dir=save_dir,
            seed=seed,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            reverb_level=reverb_level,
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
        num_frame: int = 128, epochs: int = 50, val_check_interval: float = 1.0, \
        plateau_per_validation: bool = False, \
        frame_rate: int = 100, num_velocity: int = 128, num_note: int = 88, \
        lr: float = 1e-4, dropout: float = 0.1, clip_gradient_norm: float = 1.0,seed: int = 1234, \
        cnn_channel: int = 4, cnn_kernel: int = 5, d: int = 256, pff_dim: int = 512, enc_layer: int = 3, \
        dec_layer: int = 3, enc_head: int = 4, dec_head: int = 4, weight_A: float = 1.0, weight_B: float = 1.0,\
        verbose: int = 1, num_workers: int=4, logger_name: str='csv', \
        logger_version: str|None=None, num_nodes: int=1, \
        devices: int|str="auto", strategy: str="auto", max_steps: int=-1, \
        fast_attention: bool=False, compile_model: bool=False, \
        ckpt_every_n_steps: int|None=2000, \
        resume_path:str|None=None, save_dir: str="./save_dir", augment: bool=False, \
        augment_asset_root: str|None=None, augment_manifest_dir: str|None=None, \
        feature_source: str="audio", reverb_level: str="rms"):
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
                val_check_interval (float):
                    How often to validate, as a fraction of an epoch. 1.0
                    (default) is once per epoch; 0.25 is four times.

                    Sony's MAESTRO run shards the training set four ways and
                    validates after each shard, so a 20-epoch run validates 80
                    times. 0.25 reproduces that. Checkpointing follows
                    validation, so it also decides how many candidates the run
                    leaves to select from afterwards: 80 rather than 20, which
                    is what the original had.

                    It does NOT change how often the learning rate scheduler
                    fires, which is the other half of that cadence and is not
                    settable from here. HFT.configure_optimizers declares the
                    ReduceLROnPlateau with interval="epoch", and Lightning only
                    steps a scheduler whose declared interval matches the
                    update it is doing -- so the scheduler steps once per epoch
                    on the latest validation value however often validation
                    ran. Measured: at 0.25 over two epochs, 10 checkpoints and
                    1 scheduler step. Set plateau_per_validation to correct
                    that half too.
                plateau_per_validation (bool):
                    Retime the ReduceLROnPlateau to step once per validation,
                    as Sony's loop does, rather than once per epoch. Default
                    False, which is the historical behaviour.

                    Only meaningful together with val_check_interval < 1.0, and
                    only needed for MAESTRO: Sony's MAPS run validates once an
                    epoch, so this code already matched it there. Their MAESTRO
                    run shards four ways and so steps the scheduler 80 times
                    over 20 epochs against a default patience of 10; stepping
                    20 times instead leaves the learning rate effectively
                    constant for the whole run. See PlateauPerValidation in
                    models/hft/train_hft.py.
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
                logger_version (str | None):
                    Fixed subdirectory for the logger to write into, under
                    `./logs/HFT/`. Default None allocates a fresh `version_N`
                    per process. Set it to a constant for a run chained across
                    several walltimes, so every resumed job appends to one
                    directory instead of leaving one fragment each.
                num_nodes (int):
                    Number of accelerator nodes to use for distributed training. Default is 1.
                devices (int | str):
                    GPUs per node. Default "auto" takes every visible one.

                    batch_size is PER DEVICE. Two devices at batch_size=4
                    reproduce one device at batch_size=8 exactly -- all eight
                    loss terms reduce with mean() over the same element count
                    per rank, and DDP averages the gradients, so
                    mean(mean(A), mean(B)) = mean(A + B). Two devices at
                    batch_size=8 instead doubles the effective batch and halves
                    the optimizer steps per epoch, which is a hyperparameter
                    change and not a free speedup.
                strategy (str):
                    Distributed strategy. Default "auto" resolves to "ddp" once
                    more than one device is in play.
                max_steps (int):
                    Stop after this many optimizer steps. Default -1 is no
                    limit. For a timing trial, so the run ends at a known step
                    count rather than a wall clock.
                fast_attention (bool):
                    Compute attention with scaled_dot_product_attention rather
                    than an explicit softmax. Default False. Identical
                    function, but it never materialises the
                    (batch*n_frame, heads, n_bin, n_bin) weight matrix -- which
                    at batch 8 is 1.07 GB per layer, kept twice for the
                    backward. Measured ~1.3x throughput and a third of the peak
                    memory. It is not bit-identical: summation order differs.
                    On a trained checkpoint that moves the gradient less than
                    disabling TF32 does, and note F1 by 3e-5 over 10 MAESTRO
                    pieces -- see scripts/hft_variant_ab.py, which measures both
                    against those controls.

                    The attention weights returned alongside the output become a
                    zero-width placeholder, since nothing reads them. Anything
                    wanting the paper's attention maps needs the explicit path.
                compile_model (bool):
                    torch.compile the encoder and decoder. Default False.
                    Measured ~1.5x. Costs a few minutes of compilation at every
                    process start, and recompiles whenever an input shape
                    changes -- notably the short last batch of an epoch.
                ckpt_every_n_steps (int | None):
                    How often to write the rolling restart checkpoint, in
                    optimizer steps. Default 2000. None switches it off. This is
                    what a job killed at its walltime falls back to, so it caps
                    the work lost at each handoff of a chained run.
                resume_path (str | None):
                    Whether to resume training from a checkpoint. Default is None. If
                    None, it trains from scratch.
                save_dir (str):
                    Path to save results, logs and checkpoints.
                augment (bool):
                    Apply Edwards et al.'s augmentation to training excerpts on
                    the fly. Validation is never augmented. Default is False.
                augment_asset_root (str | None):
                    Directory holding `room_ir/` and `bg_noise/`. Required when
                    augment is True.
                augment_manifest_dir (str | None):
                    Overrides the packaged asset manifests. Default is None.
                reverb_level (str):
                    How the reverb stage sets its output level. "rms" (default)
                    preserves the excerpt's energy, as Kaldi's wav-reverberate
                    does by default, removing the loudness shortcut. "peak"
                    reproduces audiomentations, and therefore Edwards.
                feature_source (str):
                    "audio" (default) reads the waveform slab and computes the
                    log-mel per item; required for augmentation. "legacy_feature"
                    reads the pre-rewrite npz spectrogram store, reproducing an
                    unaugmented run against the original files.

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
            val_check_interval=val_check_interval,
            plateau_per_validation=plateau_per_validation,
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
            logger_version=logger_version,
            num_nodes=num_nodes,
            devices=devices,
            strategy=strategy,
            max_steps=max_steps,
            fast_attention=fast_attention,
            compile_model=compile_model,
            ckpt_every_n_steps=ckpt_every_n_steps,
            resume_path=resume_path,
            save_dir=save_dir,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            feature_source=feature_source,
            reverb_level=reverb_level
        )
        hft_train.main(config)

    def train_transkun(self, base_path: str="./data/transkun/", batch_size: int = 4, epochs: int = 1000000,\
        sample_rate: float = 44100, num_workers: int=4, logger_name: str='csv', num_nodes: int=1, 
        val_steps: int=495, freq: int=3000, nProcess: int=1, resume_path:str|None=None, save_dir: str="./save_dir", \
        seed: float|None=None, augment: bool=False, augment_asset_root: str|None=None, \
        augment_manifest_dir: str|None=None, reverb_level: str="rms"):
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
                seed (float | None):
                    Seeds the chunk dithering and the per-excerpt augmentation
                    draw. Default None keeps the historical behaviour of seeding
                    from the wall clock, which means neither is reproducible
                    across runs; pass a fixed value to make both so.
                augment (bool):
                    Apply Edwards et al.'s augmentation to training excerpts on
                    the fly. Validation is never augmented. Default is False.
                augment_asset_root (str | None):
                    Directory holding room_ir/ and bg_noise/. Required when
                    augment is True.
                augment_manifest_dir (str | None):
                    Overrides the packaged asset manifests. Default is None.
                reverb_level (str):
                    How the reverb stage sets its output level. "rms" preserves
                    the excerpt's energy, as Kaldi's wav-reverberate does,
                    removing the loudness shortcut; "peak" reproduces
                    audiomentations and therefore Edwards. Default is "rms".

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
            seed=time.time() if seed is None else seed,
            sample_rate=sample_rate,
            freq=freq,
            num_workers=num_workers,
            logger_name=logger_name,
            num_nodes=num_nodes,
            resume_path=resume_path,
            val_steps=val_steps,
            nProcess=nProcess,
            save_dir=save_dir,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            reverb_level=reverb_level
        )
        transkun_train.main(config)
    
    def train_hpp(self, base_path: str="./data/hpp/", model_type: str="sp", batch_size=4, lr=0.0006, iterations: int=1000000,\
        sequence_length: int=327680, seed: int=42, sample_rate: int=16000, \
        bins_per_semitone: int = 4, hop_length: int=320, learning_rate_decay_rate=0.98, \
        learning_rate_decay_steps=10000, clip_gradient_norm=3, \
        pitch_offset: int = 21, num_workers: int=4, num_nodes: int=1, \
        logger_name: str='csv', resume_path:str|None=None, \
        save_dir: str="./save_dir", augment: bool=False, augment_asset_root: str|None=None, \
        augment_manifest_dir: str|None=None, reverb_level: str="rms", \
        checkpoint_every_n_steps: int|None=None, early_stopping: bool=True):
        """
            Train HPP with specified configuration.

            Parameters
            ----------
                base_path (str):
                    Path to extracted training data
                model_type (str):
                    Supported types are 'sp', 'base', 'tiny', 'ultra-tiny'.
                batch_size (int):
                    Batch size for training/validation. Default is 4.
                iterations (int):
                    Ceiling on Lightning's global_step, not on batches. HPPNet
                    optimises each subnet separately, so global_step advances
                    once per subnet per batch: 'sp' trains two subnets and
                    therefore consumes two steps a batch, while the
                    single-subnet variants consume one. The default of 1000000
                    is 500k batches under 'sp', the top of the 200k-500k range
                    the paper reports; the variants override it to 500000 to
                    land on the same batch count. Early stopping is what ends
                    a run, so this is a ceiling rather than a target.
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
                augment (bool):
                    Apply Edwards et al.'s augmentation to training excerpts on
                    the fly. Validation is never augmented. Default is False.
                augment_asset_root (str | None):
                    Directory holding room_ir/ and bg_noise/. Required when
                    augment is True.
                augment_manifest_dir (str | None):
                    Overrides the packaged asset manifests. Default is None.
                reverb_level (str):
                    How the reverb stage sets its output level. "rms" preserves
                    the excerpt's energy, as Kaldi's wav-reverberate does,
                    removing the loudness shortcut; "peak" reproduces
                    audiomentations and therefore Edwards. Default is "rms".
                checkpoint_every_n_steps (int | None):
                    Dump a checkpoint every this many global steps and keep all
                    of them, instead of ranking on val_loss/all and keeping the
                    best five. None keeps the ranked behaviour. Use this when
                    the model is to be picked afterwards on a decoded metric:
                    val_loss/all is ~70% frame loss, so ranking on it throws
                    away the candidates a note-F1 selection would want to see.
                early_stopping (bool):
                    Stop once val_loss/all has not improved for 25 validation
                    checks. Default is True. False trains to a fixed budget,
                    which is what the reference implementation does.


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
            save_dir=save_dir,
            augment=augment,
            augment_asset_root=augment_asset_root,
            augment_manifest_dir=augment_manifest_dir,
            reverb_level=reverb_level,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            early_stopping=early_stopping
        )

        if model_type == "sp":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet', 'frame_subnet']
            config["onset_subnet_heads"] = ['onset']
            config["frame_subnet_heads"]= ['frame', 'offset', 'velocity']
            config["fixed_dilation"] = 24
            config["model_size"] = 128
        # The variants below train one subnet, so global_step advances once per
        # batch rather than twice as it does under 'sp'. Halving the ceiling
        # keeps every variant on the same 500k batches; these overrides shadow
        # whatever the caller passed for iterations and batch_size.
        elif model_type == "base":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 128
            config["iterations"] = 500000
        elif model_type == "tiny":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 64
            config["iterations"] = 500000
        elif model_type == "ultra-tiny":
            config["SUBNETS_TO_TRAIN"] = ['onset_subnet']
            config["onset_subnet_heads"] = ['onset', 'frame', 'offset', 'velocity']
            config["frame_subnet_heads"]= []
            config["batch_size"] = 4
            config["model_size"] = 48
            config["iterations"] = 500000
        else:
            raise RuntimeError(f"Mode type: {model_type} not supported!")
        hpp_train.main(config)
