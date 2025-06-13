# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/onsets_and_frames"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.onsets_and_frames.train_onsets_and_frames --config_path \
"${BASE_DIR}/confs/onsets_and_frames.json" \
 > "${BASE_DIR}/logs/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid.txt"
 tail -n 1 -f "${BASE_DIR}/logs/logfile.log"

