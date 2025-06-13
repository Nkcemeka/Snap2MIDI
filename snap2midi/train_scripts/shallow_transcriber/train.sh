# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/shallow_transcriber"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.shallow_transcriber.train_shallow --config_path \
"${BASE_DIR}/confs/shallow.json" \
 > "${BASE_DIR}/logs/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid.txt"
 tail -n 1 -f "${BASE_DIR}/logs/logfile.log"

