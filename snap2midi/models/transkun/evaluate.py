import torch
import moduleconf
import pretty_midi
from .utilities import validateNotes, load_transkun, writeMidi, readAudio
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from snap2midi.utils.eval_mir import get_note_scores, compute_activation_metrics

@torch.no_grad()
def evaluate_test(config):
    """ 
        Evaluate the Transkun model
        based on the test set.

        Args
        ----
            config (dict): Configuration dictionary
        
        Returns
        -------
            avg_metrics_notes (dict): Note metrics
            avg_metrics_act (dict): Activation-level metrics
    """
    
    test_dir = config["test_path"]
    test_files = sorted(Path(test_dir).glob("*.pt"))
    model, conf, device = load_transkun(config["checkpoint_path"])
    results = {
        'p': [],
        'r': [],
        'f': [],
        'p_off': [],
        'r_off': [],
        'f_off': [],
        'p_act': [],
        'r_act': [],
        'f_act': [],
    }
    for each in tqdm(test_files, total=len(test_files), desc="Extracting results...."):
        with open(str(each), "rb") as f:
            obj = pickle.load(f)
        audio_path = obj["audio_filename"]

        fs, audio= readAudio(audio_path)

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

        pred_midi = writeMidi(notesEst)
        for ext in ["midi", "mid"]:
            gt_midi_path = audio_path[:-3] + "midi"
            if Path(gt_midi_path).exists:
                break
        
        assert Path(gt_midi_path).exists, f"{gt_midi_path} does not exist!"
        gt_midi = pretty_midi.PrettyMIDI(gt_midi_path)
        note_scores = get_note_scores(pred_midi, gt_midi)
        act_scores = compute_activation_metrics(pred_midi, gt_midi)

        for k in note_scores:
            results[k].append(note_scores[k])
        results['p_act'].append(act_scores[0])
        results['r_act'].append(act_scores[1])
        results['f_act'].append(act_scores[2])
    
    final_scores = {key: round(np.mean(value).item(), 3) for key, value in results.items()}
    return final_scores
