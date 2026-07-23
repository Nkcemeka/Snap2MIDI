# Import the necessary libraries
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from pathlib import Path
from typing import Optional


class HFTDivDataset(IterableDataset):
    def __init__(self, config: dict, split: str=Optional[None], shuffle: bool=False):
        super().__init__()
        self.config = config
        self.split = split
        self.shuffle = shuffle

        if split is None:
            raise ValueError("Split must be specified. Use 'train', 'val', or 'test'.")
        
        self.base_path = Path(config["base_path"])
        self.ndivs = config[f"n_div_{split}"]
        self.g = torch.Generator()
        self.g.manual_seed(self.config["seed"])   # or any integer
    
    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            div_ids = range(self.ndivs)
        else:
            # share across the workers
            div_ids = range(worker_info.id, self.ndivs, worker_info.num_workers)
        
        for div in div_ids:
            dataset = HFTDataset(self.config, div, split=self.split)
            if self.shuffle:
                indices = torch.randperm(len(dataset), generator=self.g).tolist()
            else:
                indices = range(len(dataset))
            
            for idx in indices:
                yield dataset[idx]
        
            del dataset


class HFTDataset(Dataset):
    """
        A PyTorch Dataset class for the hFT-Transformer dataset.
        This class handles the loading and processing of features and labels for training.
    """
    def __init__(self, config: dict, div: int, split: str = Optional[None]):
        """
            Initializes the HFTDataset class.

            Args:
                config (dict): Config dicionary
                div (int): Current division under consideration
                split (str): Split: train, val or test
                
        """
        # Initialize the parent class
        n_slice = config["n_slice"]
        super().__init__()

        if split is None:
            raise ValueError("Split must be specified. Use 'train', 'val', or 'test'.")

        # Stores the path so we don't load data into memory
        self.base_path = Path(config["base_path"])
        self.feature_path = self.base_path / "feature" / split / (f"dataset_feature" + str(div).zfill(3)  + ".npz")
        self.frames_path = self.base_path / "label_frames" / split / (f"dataset_label_frames" + str(div).zfill(3) + ".npz")
        self.onset_path = self.base_path / "label_onset" / split / (f"dataset_label_onset" + str(div).zfill(3) + ".npz")
        self.offset_path = self.base_path / "label_offset" / split / (f"dataset_label_offset" + str(div).zfill(3) + ".npz")
        self.velocity_path = self.base_path / "label_velocity" / split / (f"dataset_label_velocity" + str(div).zfill(3) + ".npz")
        self.idx_path = self.base_path / "idx" / split / (f"dataset_idx" + str(div).zfill(3) + ".npz")
        self.config = config

        # load the idx
        if n_slice > 1:
            idx_tmp = np.load(self.idx_path)['dataset_idx']
            idx_tmp = torch.from_numpy(idx_tmp)

            # This allows us to slice the dataset into n_slice parts
            self.idx = idx_tmp[:int(len(idx_tmp) / n_slice) * n_slice][::n_slice]
        else:
            self.idx = torch.from_numpy(np.load(self.idx_path)['dataset_idx'])
        
        # load the features and labels
        self.feature = torch.from_numpy(np.load(self.feature_path)['dataset_feature'])
        self.label_onset = torch.from_numpy(np.load(self.onset_path)['dataset_label_onset'])
        self.label_offset = torch.from_numpy(np.load(self.offset_path)['dataset_label_offset'])
        self.label_frames = torch.from_numpy(np.load(self.frames_path)['dataset_label_frames'])
        self.label_velocity = torch.from_numpy(np.load(self.velocity_path)['dataset_label_velocity'])

    def __len__(self):
        """
            Returns the length of the dataset.
        """
        return len(self.idx)

    def __getitem__(self, idx):
        """ 
            Returns the spectrogram and labels
            at a given index (idx).

            Args
            ----
                idx (int): Index value
            
            Returns
            -------
                spec : spectorgram segment
                label_onset: Onset labels 
                label_offset: Offset labels
                label_frames: Frame labels
                label_velocity: Velocity labels
        """
        # for idx_feature_s or idx_feature_start, we need to subtract the margin_b after getting our starting index
        idx_feature_s = self.idx[idx] - self.config['margin_b']
        idx_feature_e = self.idx[idx] + self.config['num_frame'] + self.config['margin_f']
        
        # idx_label_s and idx_label_e are the starting and ending indices for the labels
        idx_label_s = self.idx[idx]
        idx_label_e = self.idx[idx] + self.config['num_frame']

        # a_feature: [margin+num_frame+margin, n_feature] -(transpose)-> spec: [n_feature, margin+num_frame+margin]
        spec = (self.feature[idx_feature_s:idx_feature_e]).T

        # label_onset: [num_frame, n_note]
        label_onset = self.label_onset[idx_label_s:idx_label_e]

        # label_offset: [num_frame, n_note]
        label_offset = self.label_offset[idx_label_s:idx_label_e]

        # label_frames: [num_frame, n_note]
        # bool -> float
        label_frames = self.label_frames[idx_label_s:idx_label_e].float()

        # label_velocity: [num_frame, n_note]
        # int8 -> long
        label_velocity = self.label_velocity[idx_label_s:idx_label_e].long()
        return spec, label_onset, label_offset, label_frames, label_velocity
