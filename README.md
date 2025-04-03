# Snap2MIDI
Snap2MIDI is an implementation that takes a "short snap" of audio and converts it to MIDI. Doint this allows for the quick training of Automatic Music Transcription architectures without having to write a lot of code. By training a small model on small snapshots of audio, you can quickly verify if an architecture will perform well when scaled.

## Extraction
To extract the audio snapshots/segments and the corresponding labels, go to the extraction folder and use the following code as a guide:

```
python maestro_extract.py --args.load "./confs/maestro_conf.yml"
```

