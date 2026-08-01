import snap2midi as s2m

# HPPNet is trained on MAESTRO in the paper, and _HPPMode inherits _OAFV2Mode's
# extraction wholesale: one uncompressed .npz per track holding the 16 kHz mono
# audio plus frame-level label and velocity rolls. Nothing is windowed here --
# the dataset slices excerpts at load time -- so the store is roughly the audio
# (~46 GB as float32) plus two 88-pitch rolls at 50 fps (~25 GB).
dataset_path = "/home/simone-chieppa/Documents/maestro-v3.0.0"

extractor = s2m.extract.SnapExtractor()

# hop_length=320 at 16 kHz is the 50 fps grid HPPNet's CQT front end expects;
# it has to match train_hpp's hop_length or the labels desynchronise from the
# frames. extend_pedal follows the sustain CC, as in the OAF lineage.
extractor.extract_hpp(
    dataset_path,
    dataset_name="maestro",
    save_name="data/hpp_maestro",
)
