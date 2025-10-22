.. Snap2MIDI documentation master file, created by
   sphinx-quickstart on Wed Oct 22 10:41:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Snap2MIDI documentation
=======================

**Snap2MIDI** is a Python library that hosts state-of-the-art Audio-to-MIDI transcription systems. 
It allows you to train AI models, evaluate them, and perform inference to get MIDI transcriptions. 
The goal of this library is to provide a unified framework for researchers and developers working 
on audio-to-MIDI transcription tasks.

Installation
============

The recommended way to install Snap2MIDI (for the time being) is via pip and GitHub. 
You can install it using the following commands:

.. code-block:: bash

   git clone https://github.com/Nkcemeka/Snap2MIDI.git
   pip install .

Using Snap2MIDI
===============

After installing Snap2MIDI, you can start using it in your Python projects:

.. code-block:: python

   import snap2midi as s2m

For example, to perform data extraction for training Onsets and Frames using the MAPS dataset, 
you can use the following code:

.. code-block:: python

   from snap2midi.extract import SnapExtractor

   # Initialize dataset path
   dataset_path = "./Datasets/MAPS"
   snap_extractor = SnapExtractor()

   # Perform extraction using the MAPS dataset
   snap_extractor.extract_oaf(dataset_path, dataset_name="maps")

This creates a data directory with a folder named **oaf** that contains the extracted data 
ready for training.

To train the Onsets and Frames model using the extracted data, use the following code:

.. code-block:: python

   from snap2midi.trainer import Trainer

   trainer = Trainer()
   trainer.train_oaf()

This creates a **runs** directory with training logs and checkpoints stored in a subdirectory 
called **oaf**.

After training, perform evaluation on the test set for comparisons with other models:

.. code-block:: python

   from snap2midi.evaluator import Evaluator

   evaluator = Evaluator()
   evaluator.evaluate_oaf(checkpoint_name="checkpoint_90.pt")

You can also perform inference on your own audio files to get MIDI transcriptions:

.. code-block:: python

   from snap2midi.inference import Inference

   inference = Inference()
   inference.inference_oaf(
       audio_path="./Nanana-audio.mp3", 
       checkpoint_path="runs/oaf/checkpoint_90.pt"
   )

API Reference
=============

Snap2MIDI provides a comprehensive API reference for you to thoroughly access its functionalities without
worrying a lot about implementation details. These are the relevant API functionalities you should be aware of:

* :mod:`snap2midi.extract.SnapExtractor` - Data extraction class for various models
* :mod:`snap2midi.trainer.Trainer` - Model training class for the supported state-of-the-art Audio-to-MIDI transcription models
* :mod:`snap2midi.evaluator.Evaluator` - Model evaluation class to evaluate the models on the test sets of the extracted data
* :mod:`snap2midi.inference.Inference` - Inference class to perform MIDI transcription on audio files using trained models

.. toctree::
   :maxdepth: 2
   :caption: Snap2MIDI:

   api/index
