# Snap2MIDI
<p align="center">
  <img src="./s2m.jpg" alt="My Image" width="700">
</p>

Snap2MIDI is a library that converts audio “snapshots” into MIDI. It includes several state-of-the-art (SOTA) models such as the hFT-Transformer by Sony, Onsets and Frames by Google and Kong by Bytedance. The library is designed for ease of use, allowing users to quickly train these AMT (Automatic Music Transcription) architectures without writing much code. Its codebase is also extensible: by following the established coding paradigm, users can add new datasets and integrate additional AMT models for training.

## Installation for Development Purposes
-  First of all, clone this repo:
    ```
    git clone https://github.com/Nkcemeka/Snap2MIDI.git
    ```

-  Create a virtual environment:
    ```
    python3 -m venv venv
    ```

-  Activate the environment (for example):
    ```
    source ./venv/bin/activate
    ```

-  Go to the root directory and run the following:
    ```
    pip install -e .
    ```

## API Usage
The examples below show you how to use the Snap2MIDI API.
### Data Extraction
```
import snap2midi as s2m

# init dataset path
dataset_path = "./Datasets/MAPS"
snap_extractor = s2m.extract.SnapExtractor()

# perform extraction and use the MAPS dataset
snap_extractor.extract_oaf(dataset_path, dataset_name="maps")
```

### Training
```
import snap2midi as s2m

trainer = s2m.trainer.Trainer()
trainer.train_oaf()
```

### Evaluation
```
import snap2midi as s2m

evaluator = s2m.evaluator.Evaluator()
evaluator.evaluate_oaf(checkpoint_name="checkpoint_90.pt")
```

### Inference
```
import snap2midi as s2m

inference = s2m.inference.Inference()
inference.inference_oaf(audio_path="./Nanana-audio.mp3", checkpoint_path="runs/oaf/checkpoint_90.pt")
```

## Gradio Application
Snap2MIDI has an application interface you can play around with. 

<p align="center">
  <img src="./s2m_gradio.png" alt="Snap2MIDI Gradio Image" width="700">
</p>

To start the web server, you can leverage the API that snap2midi provides. The code below shows you how to launch the app:
```
import snap2midi as s2m

SOUNDFONT_PATH="../soundfonts/MuseScore_General.sf2" 
CHECKPOINT_PATH = "runs/kong/checkpoint_180000.pt"
CHECKPOINT_PEDAL = "runs/kong_pedal/checkpoint_180000.pt"
launcher = s2m.launch_gradio.launcher(SOUNDFONT_PATH, CHECKPOINT_PATH, CHECKPOINT_PEDAL)
```

You can download the official checkpoints from HuggingFace. You can view the HuggingFace repo here:
```
https://huggingface.co/nkcemeka/Snap2MIDI
```

