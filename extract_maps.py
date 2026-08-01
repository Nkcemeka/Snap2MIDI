import snap2midi as s2m

# init dataset path
dataset_path = "/home/simone-chieppa/Documents/maps/MAPS"
snap_extractor = s2m.extract.SnapExtractor()

# perform extraction and use the MAPS dataset
# save_name is the name of the folder to extract the data to...
snap_extractor.extract_hft(dataset_path, dataset_name="maps", save_name="data/hft_maps")