# Snap2MIDI
Snap2MIDI is a package that allows you take a "snap" of audio and convert it to MIDI. It contains different architectures including SOTA models such as OnsetsAndFrames, Kong, etc. The package is also designed such that you can quickly train these Automatic Music Transcription (AMT) architectures without having to write a lot of code. By training a small version of these models on small snapshots of audio, you can quickly verify if an architecture will perform well when scaled. The codebase is also designed such that it is extensible if you follow the coding paradigm; you should be able to add new datasets and new AMT models to train on.

## Installation
1. First of all, clone this repo:

    ```
    git clone https://github.com/Nkcemeka/Snap2MIDI.git
    ```

2. Create a virtual environment:

    ```
    python3 -m venv venv
    ```

3. Activate the environment (for example):

    ```
    source ./venv/bin/activate
    ```

4. Go to the root directory and run the following:

    ```
    pip install -e .
    ```

## Extraction
To extract the audio snapshots/segments and the corresponding labels for a dataset of your choice, go to the extractors folder and run the necessary code using the following below as a guide (note that the configuration can be used for any dataset; currently, the extraction code supports MAESTRO, GuitarSet, MusicNet, MAPS and Slakh). Also be aware of the different modes for extraction; for example, the HfT mode allows you to extract your features based on the hFT-Transformer architecture design. This 'modal feature' was done to accomodate modes that might not follow the general narrative:

```
python -u snap_extract.py --mode="general" --config_path="./confs/snap_conf.json"
```

You can also edit and run the extract.sh file.

## Training
To train an architecture (e.g OnsetsAndFrames), you can go to the train_scripts/onsets_and_frames folder
 and run the corresponding training bash script. E.g:
```
./train.sh
```

## Inference
To perform inference for any architecture, go into the relevant folder and run the following:
```
./inference.sh <audio_filename> <output_name> <feature_str>
```

## Model Directory Structure
Each model has a directory structure similar to that below! The confs directory allows you to create different config files to suit your purpose. In the datasets directory, you can also create different dataset classes. The logs file contains your model's logs. It also contains the process id of the training session.
```
.
├── confs
│   ├── training_config.json
│   └── inference_config.json
├── datasets
│   ├── dataset_oaf.py
├── inference.sh
├── __init__.py
├── logs
│   ├── logfile.log
│   └── pid.txt
├── model.py
├── model_inference.py
├── train_model.py
└── train.sh
```

## Gradio Application
Snap2MIDI now has an application interface. After training any of the models above, you can run the gradio application to access a simple user interface and test them out. To run the application, go to the gradio_app directory and run the ./run.sh file. Make sure you update the file with the correct path to the soundfont file. Alternatively, you can try:
```
python app.py --soundfont_path=<SOUNDFONT_PATH>
```
