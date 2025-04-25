# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/onsets_and_frames"
nohup python -m snap2midi.train_scripts.onsets_and_frames.train_onsets_and_frames --config_path \
"${BASE_DIR}/confs/onsets_and_frames.json" \
 > "${BASE_DIR}/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/pid.txt"

