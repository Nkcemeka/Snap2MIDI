import snap2midi as s2m

# Moved out of ~/Downloads deliberately: extract_kong stores only the audio and
# writes the MIDI path into each h5 as an absolute reference, so the extracted
# dataset keeps pointing back here for its labels.
dataset_path = "/home/simone-chieppa/Documents/maestro-v3.0.0"

extractor = s2m.extract.SnapExtractor()

# Defaults match Edwards et al.: Kong's architecture on 10 s training examples
# at 16 kHz. hop_size=1.0 gives one excerpt per second of audio, so the windows
# overlap 10x -- they are indexed, not materialised, so the store is just the
# resampled audio (~23 GB for MAESTRO's 198.7 h at int16 mono 16 kHz).
extractor.extract_kong(
    dataset_path,
    dataset_name="maestro",
    save_name="data/kong_maestro",
)
