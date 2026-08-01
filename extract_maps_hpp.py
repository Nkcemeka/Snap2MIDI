import snap2midi as s2m

# The point of this store is the test split. HPPNet's Table 4 trains on MAESTRO
# v3 and evaluates on MAPS, which is the cross-domain setting Edwards-style
# augmentation is supposed to help; the published no-augmentation reference
# there is frame F1 87.56 / note F1 86.63. _get_maps_train_val_test reserves
# ENSTDkAm and ENSTDkCl -- the real Disklavier recordings -- as test, so that
# comparison needs no MAPS training data at all.
#
# The train/val splits come out of the same pass and cost little (270 MUS files,
# ~17 h of audio, so roughly 6 GB once resampled to 16 kHz with the two label
# rolls). They are extracted rather than skipped so a MAPS-trained arm is
# possible later without re-running this.
dataset_path = "/home/simone-chieppa/Documents/maps/MAPS"

extractor = s2m.extract.SnapExtractor()

# Same grid as the MAESTRO store: hop 320 at 16 kHz is the 50 fps that HPPNet's
# CQT front end expects, and it has to match whatever train_hpp was given or the
# labels drift against the frames.
extractor.extract_hpp(
    dataset_path,
    dataset_name="maps",
    save_name="data/hpp_maps",
)
