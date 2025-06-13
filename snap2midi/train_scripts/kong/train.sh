# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/kong"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.kong.train_kong --config_path \
"${BASE_DIR}/confs/kong_config.json" \
 > "${BASE_DIR}/logs/logfile_kong.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid_kong.txt"
tail -n 1 -f "${BASE_DIR}/logs/logfile_kong.log"


