# This should be run from the extractors folder. Move it there.
#nohup python -u snap_extract.py --mode="general" --config_path="./confs/snap_conf.json" > logfile.log 2>&1 & echo $! > pid.txt
nohup python -u snap_extract.py --mode="hft" --config_path="./confs/hft_config.json" > logfile.log 2>&1 & echo $! > pid.txt
