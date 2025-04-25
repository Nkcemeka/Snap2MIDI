# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/shallow_transcriber"
nohup python -m snap2midi.train_scripts.shallow_transcriber.train_shallow --config_path \
"${BASE_DIR}/confs/shallow.json" \
 > "${BASE_DIR}/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/pid.txt"

