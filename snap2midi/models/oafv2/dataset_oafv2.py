# Imports
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Sequence, Optional

class OAFV2Dataset(Dataset):
    def __init__(self, config: dict, emb_paths: Optional[list[str]]=None) -> None:
        """
            Dataset class to load training segments for
            the Onsets and Frames model V2.

        Args
        ----
            emb_paths (list): 
                List of paths to npz files containing data for training.
        """
        super().__init__()
        if emb_paths is None:
            raise ValueError(f"{self.__class__.__name__} needs path to embeddings!")

        self.sequence_length = config["sequence_length"]
        self.random = np.random.RandomState(config["seed"])
        self.hop_length = config["hop_length"]
        self.data = [] # path to files
        for emb_path in emb_paths:
            assert Path(emb_path).exists(), f"{emb_path} does not exist."
            self.emb_path = emb_path
            self.data.extend(sorted(Path(emb_path).glob("*.npz")))

    def __getitem__(self, index: int) -> Sequence[np.ndarray]:
        """ 
            Gets item from dataset based on index.

            Args
            ----
                index (int): Index 
            
            Returns
            -------
                None
        """
        item_path = str(self.data[index])
        item = np.load(item_path)
        result = {}

        if self.sequence_length is not None:
            audio_length = len(item["audio"])
            frame_start = self.random.randint(audio_length - self.sequence_length) // self.hop_length
            n_frames = self.sequence_length // self.hop_length
            frame_end = frame_start + n_frames

            start_samples = frame_start * self.hop_length
            end_samples = start_samples + self.sequence_length

            result["audio"] = item["audio"][start_samples:end_samples].astype(np.float32)
            result["label"] = item["label"][frame_start:frame_end, :]
            result["velocity"] = item["velocity"][frame_start:frame_end, :]
        else:
            result["audio"] = item["audio"].astype(np.float32)
            result["label"] = item["label"]
            result["velocity"] = item["velocity"]
        
        result['onset'] = (result['label'] == 3).astype(np.float32)
        result['offset'] = (result['label'] == 1).astype(np.float32)
        result['frame'] = (result['label'] > 1).astype(np.float32)
        result['velocity'] - (result["velocity"]).astype(np.float32)
        return result

    def __len__(self) -> int:
        """ 
            Length of dataset.

            Args
            ----
                length (int): Length of data
        """
        return len(self.data)

if __name__ == "__main__":
    dataset = OAFV2Dataset(["./extractors/maestro_events_segments/train/"])
    print(dataset[0][0].shape, dataset[0][1].shape)
