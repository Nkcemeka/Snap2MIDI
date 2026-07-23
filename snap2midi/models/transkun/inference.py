import torch
import moduleconf
import pretty_midi
from .utilities import validateNotes, writeMidi, readAudio
import numpy as np
from pathlib import Path


def trans(audioPath: str, \
          weight: str):
    """ 
        Transcribes an audio file using the transkun model

        Args:
            audioPath (str): path to audio file
            weight (str): path to pretrained weights
        
        Returns:
            None
    """
    confPath = str(Path(__file__).parent / "conf.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confManager = moduleconf.parseFromFile(confPath)
    
    # For this to work, change the 
    TransKun = confManager["Model"].module.Transkun
    conf = confManager["Model"].config

    checkpoint = torch.load(weight, map_location = device)

    model = TransKun(conf = conf).to(device)

    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()
    torch.set_grad_enabled(False)

    fs, audio= readAudio(audioPath)

    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
    )

    x = torch.from_numpy(audio).to(device)

    notesEst = model.transcribe(x, stepInSecond=conf.segmentHopSizeInSecond, \
        segmentSizeInSecond=conf.segmentSizeInSecond, discardSecondHalf=False)

    outputMidi = writeMidi(notesEst)
    return outputMidi

def inference(config: dict):
    """ 
        Perform inference

        Args
        ----
            config (dict): Config dictionary
        
        Returns
        -------
            midi_obj (pretty_midi.PrettyMIDI): PrettyMIDI object.
    """
    filename = config["filename"]
    audio_path = config["audio_path"]
    weight = config["checkpoint_path"]
    midi_obj = trans(audio_path, weight)
    if filename is None:
        return midi_obj
    midi_obj.write(f'{filename}.mid')
