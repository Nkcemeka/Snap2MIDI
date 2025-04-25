# This should be run from the extractors folder. Move it there.
nohup python maestro_events_extract.py --args.load "./confs/maestro_events_conf.yml" > logfile.log 2>&1 & echo $! > pid.txt
