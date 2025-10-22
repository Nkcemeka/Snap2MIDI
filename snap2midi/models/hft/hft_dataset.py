# Import the necessary libraries
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

class HFTDataset(Dataset):
    """
        A PyTorch Dataset class for the hFT-Transformer dataset.
        This class handles the loading and processing of features and labels for training.
    """
    def __init__(self, base_path: str, config: dict, split: str = Optional[None]):
        """
            Initializes the HFTDataset class.

            Args:
                
        """
        # Initialize the parent class
        n_slice = config["n_slice"]
        super().__init__()

        if split is None:
            raise ValueError("Split must be specified. Use 'train', 'val', or 'test'.")

        # Stores the path so we don't load data into memory
        self.base_path = Path(base_path)
        self.feature_path = self.base_path / "feature" / split / "dataset_feature.npz"
        self.frames_path = self.base_path / "label_frames" / split / "dataset_label_frames.npz"
        self.onset_path = self.base_path / "label_onset" / split / "dataset_label_onset.npz"
        self.offset_path = self.base_path / "label_offset" / split / "dataset_label_offset.npz"
        self.velocity_path = self.base_path / "label_velocity" / split / "dataset_label_velocity.npz"
        self.idx_path = self.base_path / "idx" / split / "dataset_idx.npz"
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
    

if __name__ == "__main__":
    # Example usage
    config = {
        'input': {
            'num_frame': 128,
            'margin_b': 32,
            'margin_f': 32
        },
        'midi': {
            'num_note': 128
        },
        'feature': {
            'log_offset': 0.1
        }
    }

    base_path = '/home/nkcemeka/Documents/snap/snap2midi/extractors/maps_segments'
    dataset = HFTDataset(n_slice=16, base_path=base_path, config=config, split='train')
    print(f"Dataset length: {len(dataset)}")
    spec, onset, offset, frames, velocity = dataset[0]
    print(f"Spec shape: {spec.shape}, Onset shape: {onset.shape}")
