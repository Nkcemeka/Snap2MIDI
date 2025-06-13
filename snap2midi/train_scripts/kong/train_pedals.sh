# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/kong"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.kong.train_kong_pedals --config_path \
"${BASE_DIR}/confs/kong_pedal_config.json" \
 > "${BASE_DIR}/logs/logfile_pedal.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid_pedal.txt"
 tail -n 1 -f "${BASE_DIR}/logs/logfile_pedal.log"


