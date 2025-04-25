# Snap2MIDI
Snap2MIDI is a package that allows you take a "short snap" of audio and convert it to MIDI. Doing this allows for the quick training of Automatic Music Transcription (AMT) architectures without having to write a lot of code. By training a small model on small snapshots of audio, you can quickly verify if an architecture will perform well when scaled. The codebase is also designed such that it is extensible if you follow the coding paradigm; you should be able to add new datasets and new AMT models to train on.

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
pip install .
```

## Extraction
To extract the audio snapshots/segments and the corresponding labels for a dataset of your choice, go to the extractors folder and run the necessary code using the following below (for MAESTRO) as a guide:

```
python maestro_extract.py --args.load "./confs/maestro_conf.yml"
```

## Training
To train an architecture (e.g OnsetsAndFrames), you can go to the train_scripts/onsets_and_frames folder
 and run the corresponding training bash script. E.g:
```
./train.sh
```

