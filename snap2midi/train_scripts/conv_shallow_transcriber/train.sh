# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/conv_shallow_transcriber"
mkdir -p "${BASE_DIR}/logs"
nohup python -m snap2midi.train_scripts.conv_shallow_transcriber.train_conv --config_path \
"${BASE_DIR}/confs/conv_shallow.json" \
 > "${BASE_DIR}/logs/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/logs/pid.txt"
 tail -n 1 -f "${BASE_DIR}/logs/logfile.log"

