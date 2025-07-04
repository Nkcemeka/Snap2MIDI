"""
    Author: Chukwuemeka Nkama
    Date: 7/3/2-25
    Description: Gradio app for the Snap2Midi project.
"""
import gradio as gr
import numpy as np
from snap2midi.train_scripts.kong.kong_inference import inference as inf_kong
from snap2midi.train_scripts.onsets_and_frames.onsets_inference import inference as inf_onsets
from snap2midi.train_scripts.shallow_transcriber.shallow_inference import inference as inf_shallow
from snap2midi.train_scripts.conv_shallow_transcriber.conv_shallow_inference import inference as inf_conv_shallow
import pretty_midi
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image

# model config paths
KONG_CONFIG_PATH = "../train_scripts/kong/confs/kong_inference_config.json"
ONSETS_CONFIG_PATH = "../train_scripts/onsets_and_frames/confs/onsets_inference_config.json"
SHALLOW_CONFIG_PATH = "../train_scripts/shallow_transcriber/confs/shallow_inference_config.json"
CONV_SHALLOW_CONFIG_PATH = "../train_scripts/conv_shallow_transcriber/confs/conv_shallow_inference_config.json"

# Dictionary of paths to the model config files
path_dict = {
    "Kong": KONG_CONFIG_PATH,
    "OnsetsAndFrames": ONSETS_CONFIG_PATH,
    "ShallowTranscriber": SHALLOW_CONFIG_PATH,
    "Convolutional Shallow Transcriber": CONV_SHALLOW_CONFIG_PATH
}

def update_btn_visibility():
    """
        Update the visibility of a button in the Gradio interface.
    """
    return gr.update(visible=False)
    

def stretch_height(img, value=0):
    """    
        Stretch the height of the image to the target height while maintaining aspect ratio.
        Args:
            img (np.ndarray): Input image array.
            value (int, optional): Value to fill the resized image. Defaults to 0.
        
        Returns:
            np.ndarray: Resized image array.
    """
    pil_img = Image.fromarray(img)
    w, h = pil_img.size
    resized_img = np.array(pil_img.resize((w, w//4)))
    mask = resized_img > 0
    resized_img[mask] = value 

    # these libraries take origins as left-top, so we flip the image
    resized_img = np.flipud(resized_img)
    return np.array(resized_img)

def run_inference(model_type: str, filepath: str, config: dict) -> pretty_midi.PrettyMIDI:
    """
        Run inference on the audio file using the specified model type.

        Args:
            model_type (str): Type of the model to use for inference.
            filepath (str): Path to the audio file.
            config (dict): Configuration dictionary for the model.
        
        Returns:
            pretty_midi.PrettyMIDI: Processed MIDI object.
    """
    if model_type == "Kong":
        return inf_kong(
            audio_path=filepath,
            config=config,
            feature_str=config["feature_str"],
            filename=None
        )
    elif model_type == "OnsetsAndFrames":
        return inf_onsets(
            audio_path=filepath,
            config=config,
            feature_str=config["feature_str"],
            filename=None
        )
    elif model_type == "ShallowTranscriber":
        return inf_shallow(
            audio_path=filepath,
            config=config,
            feature_str=config["feature_str"],
            filename=None
        )
    elif model_type == "Convolutional Shallow Transcriber":
        return inf_conv_shallow(
            audio_path=filepath,
            config=config,
            feature_str=config["feature_str"],
            filename=None
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def process_audio(filepath: str, config: dict,\
                  soundfont_path: str, model_type: str) -> tuple[pretty_midi.PrettyMIDI, np.ndarray]:
    """
        Process the audio file and 
        return the processed audio.

        Args:
            filepath (str): Path to the audio file.
            config (dict): Configuration dictionary for the model.
            model_type (str): Type of the model to use for inference.
            soundfont_path (str): Path to the soundfont file.
        
        Returns:
            pretty_midi.PrettyMIDI: Processed MIDI object.
            audio (np.ndarray): MIDI audio data.
    """
    midi_obj = run_inference(model_type, filepath, config)

    audio_data = midi_obj.fluidsynth(fs=config["sample_rate"],
        sf2_path=f"{soundfont_path}")
    
    return (config["sample_rate"], audio_data), midi_obj

def get_piano_roll(midi_obj: pretty_midi.PrettyMIDI, config):
    """
        Convert the MIDI object to a piano roll image.

        Args:
            midi_obj (pretty_midi.PrettyMIDI): MIDI object to convert.
            config (dict): Configuration dictionary containing frame rate.
        
        Returns:
            np.ndarray: Piano roll image.
    """
    fs = config["frame_rate"]
    piano_roll = midi_obj.get_piano_roll(fs=int(fs))
    # Normalize grayscale to [0,1]
    gray_norm = piano_roll / 255.0
    
    # Create red tint: red channel = inv, green and blue = 0
    red_channel = (gray_norm * 255).astype(np.uint8)
    green_channel = np.zeros_like(red_channel)
    blue_channel = np.zeros_like(red_channel)

    red_channel = stretch_height(red_channel, value=255)
    green_channel = stretch_height(green_channel)
    blue_channel = stretch_height(blue_channel)
    
    # Combine channels
    colored_img = np.stack([red_channel, green_channel, blue_channel], axis=2)
    
    mask_background = red_channel == 0  # tweak threshold as needed
    colored_img[mask_background] = [255, 255, 255]
    
    return gr.Image(value=colored_img, visible=True, image_mode="RGB",\
                     label="Piano Roll Image", show_download_button=True)


def save_midi(midi_obj: pretty_midi.PrettyMIDI, filename: str):
    """
        Save the MIDI object to a file.

        Args:
            midi_obj (pretty_midi.PrettyMIDI): MIDI object to save.
            filename (str): Name of the file to save the MIDI object to.
    """
    midi_obj.write(f"{filename}.mid")

def load_config(config_path: str) -> dict:
    """
        Load the configuration for the specified model type.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Configuration dictionary for the specified model type.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Snap2Midi Gradio app.")
    parser.add_argument("--soundfont_path", type=str, required=True, help="Path to the soundfont file")
    args = parser.parse_args()

    # Set soundfont path
    soundfont_path = args.soundfont_path

    with gr.Blocks(theme="gstaff/sketch") as demo:
        config = gr.State()
        gr.Markdown(
            """
            # Snap2Midi Gradio App
            """
        )

        with gr.Row():
            with gr.Group():
                model_type = gr.Dropdown(
                    choices=[
                        "Kong",
                        "OnsetsAndFrames",
                        "ShallowTranscriber",
                        "Convolutional Shallow Transcriber",
                    ],
                    label="Choose your AI Music Transcription Model",
                )
                    
                input_audio = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Input Audio",
                )
        
        with gr.Row():
            with gr.Tabs():
                with gr.Tab("Transcribe Audio to MIDI"):
                    with gr.Group():
                        piano_roll_img = gr.Image(label="Piano Roll Image", image_mode="RGB",\
                                                show_download_button=True, visible=False) 
                        out = gr.Audio(label="Processed MIDI Audio", show_download_button=True)
                        transcribe_btn = gr.Button("Transcribe", size="md")
                
                with gr.Tab("Download MIDI"):
                    with gr.Group():
                        midi_filename = gr.Textbox(label="MIDI File Name", \
                                                placeholder="Type MIDI file name and click the Download button...", lines=1, max_lines=1)
                        download_midi_btn = gr.Button("Download MIDI", size="md")
                    
        
            
            
        midi_state = gr.State() # To store midi transcription

        # Set the model type and load the corresponding config
        model_type.change(lambda x: load_config(path_dict[x]), inputs=[model_type], outputs=[config])
        demo.load(lambda x: load_config(path_dict[x]), inputs=[model_type], outputs=[config])

        # Set click events to make transcription possible on clicking btn and making btn invisible
        # after that
        transcribe_btn.click(process_audio, inputs=[input_audio, config, \
                            gr.State(soundfont_path), model_type], outputs=[out, midi_state])
        transcribe_btn.click(update_btn_visibility, inputs=[], outputs=[transcribe_btn])

        # handler events for generating a MIDI piano roll on transcription
        out.change(get_piano_roll, inputs=[midi_state, config], outputs=[piano_roll_img])
        
        # Save MIDI file and download it
        midi_filename_value = gr.State()
        midi_filename.change(lambda x: x if x else "transcription", inputs=[midi_filename],\
                              outputs=[midi_filename_value])
        download_midi_btn.click(save_midi, inputs=[midi_state, midi_filename_value], outputs=None)


    demo.launch(debug=True)