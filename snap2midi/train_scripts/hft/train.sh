# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/hft"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.hft.train_hft --config_path \
"${BASE_DIR}/confs/train_hft_config.json" \
 > "${BASE_DIR}/logs/logfile_hft.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid_hft.txt"
tail -n 1 -f "${BASE_DIR}/logs/logfile_hft.log"


