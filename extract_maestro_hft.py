"""hFT-Transformer extraction of MAESTRO, so HFT can run Kong's protocol.

The only HFT store on disk is data/hft_maps, which is MAPS. Kong trains on
MAESTRO and is tested out of domain on MAPS, and HFT cannot be put on that axis
without a MAESTRO extraction -- this is it. Same source tree Kong reads
(maestro-v3.0.0) and the extractor's own defaults, so nothing about the labels
or the front end differs from what extract_maps.py produced for MAPS.

n_div_* are left at 1 deliberately, and it is not a free parameter. HFTDataset
takes the division count from the *training* config -- `self.ndivs =
config[f"n_div_{split}"]` -- not from the store, and its cross-division meta
check only runs over range(1, ndivs), so it verifies nothing when ndivs is 1.
Extract at 8 and then train at the trainer's default of 1 and the run silently
reads division 0 alone: one eighth of MAESTRO, no error, a plausible loss curve.
Keeping the store at 1 makes that mismatch unrepresentable. The old reason to
divide -- the IterableDataset sharded workers by division, so n_div=1 left three
of four workers idle -- died with the map-style rewrite in hft_dataset.py.

Sizing, measured against data/hft_maps (18.3 h of audio, 11 GB of slabs, so
0.60 GB/h) and scaled to MAESTRO's 198.7 h:

    final store   ~132 GB   119 GB of slabs, plus the ~13 GB of per-track test
                            npz that _collate_hft_split_div keeps on purpose --
                            it deletes them for train and val only.
    peak on disk  ~226 GB   _extract_hft writes every split's npz before any
                            collation starts, so all 131 GB of them are live
                            while the 95 GB train slab is built beside them.
                            Deletion happens per division, i.e. after the whole
                            train slab at n_div=1.

414 GB were free when this was written, so the peak leaves ~190 GB. Check df
before re-running it on a fuller disk.

Expect 1.5-3 h: MAPS took 8 minutes for 18.3 h, MAESTRO is 10.9x that, and
writing 226 GB is not free on top.

Usage:
    python extract_maestro_hft.py
"""

import snap2midi as s2m

# Same tree extract_maestro_kong.py reads. Kept out of ~/Downloads because the
# extracted store references it for labels.
DATASET_PATH = "/home/simone-chieppa/Documents/maestro-v3.0.0"


def main() -> int:
    s2m.extract.SnapExtractor().extract_hft(
        DATASET_PATH,
        dataset_name="maestro",
        save_name="data/hft_maestro",
        # See the module docstring: 1 is what the trainer defaults to, and a
        # mismatch here is silent.
        n_div_train=1,
        n_div_val=1,
        n_div_test=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
