import snap2midi as s2m

dataset_path = "/home/simone-chieppa/Documents/maps/MAPS"

extractor = s2m.extract.SnapExtractor()
extractor.extract_hft(dataset_path, dataset_name="maps", save_name="data/hft_maps")
