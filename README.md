# Snap2MIDI
Snap2MIDI is an implementation that takes a "short snap" of audio and converts it to MIDI. Doing this allows for the quick training of Automatic Music Transcription (AMT) architectures without having to write a lot of code. By training a small model on small snapshots of audio, you can quickly verify if an architecture will perform well when scaled. The codebase is also designed such that it is extensible if you follow the coding paradigm; you should be able to add new datasets and new AMT models to train on.

## Extraction
To extract the audio snapshots/segments and the corresponding labels for a dataset of your choice, go to the extractors folder and run the necessary code using the following below (for MAESTRO) as a guide:

```
python maestro_extract.py --args.load "./confs/maestro_conf.yml"
```

