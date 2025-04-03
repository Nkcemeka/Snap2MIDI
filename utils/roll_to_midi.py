"""
    File: roll_to_midi.py
    Author: Chukwuemeka L. Nkama
    Date: 4/3/2025
    Description: Converts piano roll to MIDI file
"""

import numpy as np
import pretty_midi
import argparse


def conv_to_midi(piano_roll, output_path, filename, frame_rate):
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
    duration = piano_roll.shape[0]/frame_rate # duration of piano roll
    midi = pretty_midi.PrettyMIDI()
    prog = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=prog)
    for i, frame in enumerate(piano_roll):
        # Get all the pitches that are active
        pitches = np.where(frame==1)[0]
        for pitch in pitches:
            note = pretty_midi.Note( 
                velocity=100, pitch=pitch, start=(i/frame_rate), 
                end=(i/frame_rate) + duration
            )
            piano.notes.append(note)
    midi.instruments.append(piano)
    midi.write(f"{output_path}/{filename}.mid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--roll_path')
    parser.add_argument('--output_path')
    parser.add_argument('--frame_rate')
    parser.add_argument('--filename')

    args = parser.parse_args()
    roll = np.load(args.roll_path)["roll"]
    conv_to_midi(roll, args.output_path, args.filename, args.frame_rate)
