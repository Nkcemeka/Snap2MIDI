# This should be run from the extractors folder. Move it there.
nohup python snap_extract.py --args.load "./confs/snap_conf.yml" > logfile.log 2>&1 & echo $! > pid.txt
