import snap2midi as s2m

# MAPS is the out-of-distribution evaluation set: the model never sees it in
# training, and it is where Edwards measures the +4.0 F1 that augmentation buys.
# _get_maps_train_val_test puts the ENSTDkAm and ENSTDkCl (Disklavier) subsets in
# test, which is the standard MAPS test configuration the paper reports on.
dataset_path = "/home/simone-chieppa/Documents/maps/MAPS"

extractor = s2m.extract.SnapExtractor()
extractor.extract_kong(
    dataset_path,
    dataset_name="maps",
    save_name="data/kong_maps",
)
