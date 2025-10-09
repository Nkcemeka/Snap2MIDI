import numpy as np
import pretty_midi
import argparse
from pathlib import Path

def conv_to_midi(piano_roll: np.ndarray, filename: str, frame_rate: int, \
                 output_path: str|None=None, save: bool=False) -> pretty_midi.PrettyMIDI:
    """ 
        Args:
            piano_roll (np.ndarray): Piano roll of shape (frames, 128)
            output_path (str): Path to store MIDI file
            filename (str): name of MIDI file
            frame_rate (int): frame_rate of roll. Use the original or 
                          the final output will be slowed down or sped up
                          compared to original!
        
        Returns:
            midi (MIDI object): MIDI file
    """
    piano_roll = piano_roll.astype(int) # convert to int
    midi = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)

    for pitch in range(128):
        prev_val = 0.0 # stores velocity value of previous frame [0, 1]
        for frame in range(piano_roll.shape[0]):
            if piano_roll[frame, pitch] > 0.0 and prev_val == 0.0:
                # Note on
                start = frame / frame_rate
                vel = piano_roll[frame, pitch]
                prev_val = piano_roll[frame, pitch]
            elif piano_roll[frame, pitch] == 0.0 and prev_val > 0.0:
                # Note off
                end = frame / frame_rate

                note = pretty_midi.Note(
                    velocity=vel, pitch=pitch, start=start, end=end
                )
                piano.notes.append(note)
                prev_val = 0
            else:
                continue
    midi.instruments.append(piano)

    if save:
        # Save the MIDI file
        if output_path is None:
            raise ValueError("output_path cannot be None if save is True!")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        midi.write(f"{output_path}/{filename}.mid")
    else:
        return midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--roll_path')
    parser.add_argument('--output_path')
    parser.add_argument('--frame_rate')
    parser.add_argument('--filename')

    args = parser.parse_args()
    roll = np.load(args.roll_path)["roll"]
    conv_to_midi(roll, args.output_path, args.filename, args.frame_rate)
