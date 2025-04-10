# This should be run from the extractors folder. Move it there.
nohup python maestro_extract.py --args.load "./confs/maestro_conf.yml" > logfile.log 2>&1 &
