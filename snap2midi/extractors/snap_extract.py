"""
File: snap_extract.py
Author: Chukwuemeka L. Nkama
Date: 2025-02-05

Description: This file extracts audio segments, features and labels for a collection 
             of audio and MIDI files.
"""

# Imports
from modes import GeneralMode, HFTMode
import argparse
import json

class SnapExtractor:
    """
        Class to extract audio segments, features and labels 
         from any given dataset. This assumes that we have
        two lists of files, one for audio and the other for MIDI.
    """

    def __init__(self, config: dict, mode: str="general") -> None:
        """
            Args:
                config (dict): Configuration dictionary containing the parameters
                mode (str): Mode of extraction. Default is "general".

            Returns:
                None
        """
        if mode == "general":
            GeneralMode(config)
        elif mode == "hft":
            HFTMode(config)
        else:
            raise ValueError(f"Mode {mode} not supported! \
                    Supported modes are: general, hft")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, \
                        help='Path to the config file')
    parser.add_argument('--mode', type=str, default='general', choices=['general', 'hft', '...'], \
                        help='Mode of operation.')
    args = parser.parse_args()
    mode = args.mode
    config = args.config_path

    # load JSON file
    with open(args.config_path, 'r') as filename:
        content = filename.read()

    # parse JSON file
    config = json.loads(content)
    SnapExtractor(config, mode=mode)