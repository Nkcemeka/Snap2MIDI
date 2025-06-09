# set base_directory
BASE_DIR="/home/nkcemeka/Documents/snap/snap2midi/train_scripts/kong"
nohup python -m snap2midi.train_scripts.kong.train_kong_pedals --config_path \
"${BASE_DIR}/kong_pedal_config.json" \
 > "${BASE_DIR}/logfile.log" 2>&1 & echo $! > "${BASE_DIR}/pid.txt"


