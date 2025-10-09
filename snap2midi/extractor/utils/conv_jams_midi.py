import numpy as np
import pretty_midi
import jams

def jams_to_midi(jam: jams.JAMS, q: int = 1) -> pretty_midi.PrettyMIDI:
    """
        Convert jams to midi using pretty_midi.
        Gotten from the `marl repo`_.
        .. _marl repo: https://github.com/marl/GuitarSet/blob/master/visualize/interpreter.py

        Args:
            jam (jams.JAMS): Jams object
            q (int): 1: with pitch bend. q = 0: without pitch bend.
        
        Returns:
            midi: PrettyMIDI object
    """
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    for anno in annos:
        midi_ch = pretty_midi.Instrument(program=25)
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100 + np.random.choice(range(-5, 5)),
                pitch=pitch, start=st,
                end=st + dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
        if len(midi_ch.notes) != 0:
            midi.instruments.append(midi_ch)
    return midi
