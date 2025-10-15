import gradio as gr
import numpy as np
from snap2midi import Inference
import pretty_midi
import argparse
from PIL import Image
from huggingface_hub import hf_hub_download

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

def run_inference(model_type: str, filepath: str, checkpoint_path: str, \
                  checkpoint_pedal: str|None=None) -> pretty_midi.PrettyMIDI:
    """
        Run inference on the audio file using the specified model type.

        Args:
            model_type (str): Type of the model to use for inference.
            filepath (str): Path to the audio file.
            checkpoint_path (str): Path to the model checkpoint file.
            checkpoint_pedal (str|None, optional): Path to the pedal model checkpoint file (if applicable). 
                                                   Defaults to None.
        
        Returns:
            pretty_midi.PrettyMIDI: Processed MIDI object.
    """
    inference = Inference()
    if model_type == "Kong":
        if checkpoint_pedal is None:
            raise ValueError("Pedal checkpoint path must be provided for Kong model.")
        
        user_ext_config = {'n_mels': 229, 'max_pitch': 108, 'min_pitch': 21, \
            'sample_rate': 16000, 'frame_rate': 100, 'mel_n_fft': 2048}
        return inference.inference_kong(audio_path=filepath, filename=None, checkpoint_note_path=checkpoint_path,\
                                        checkpoint_pedal_path=checkpoint_pedal, user_ext_config=user_ext_config)
    elif model_type == "OnsetsAndFrames":
        return inference.inference_oaf(audio_path=filepath, filename=None, checkpoint_path=checkpoint_path)
    elif model_type == "HFT":
        return inference.inference_hft(audio_path=filepath, filename=None, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def process_audio(filepath: str, soundfont_path: str, model_type: str, \
                  checkpoint_path: str, checkpoint_pedal: str|None=None) -> tuple[pretty_midi.PrettyMIDI, np.ndarray]:
    """
        Process the audio file and 
        return the processed audio.

        Args:
            filepath (str): Path to the audio file.
            soundfont_path (str): Path to the soundfont file.
            model_type (str): Type of the model to use for inference.
            checkpoint_path (str): Path to the model checkpoint file.
            checkpoint_pedal (str|None, optional): Path to the pedal model checkpoint file (if applicable). 
                                                   Defaults to None.
            
        Returns:
            pretty_midi.PrettyMIDI: Processed MIDI object.
            audio (np.ndarray): MIDI audio data.
    """
    midi_obj = run_inference(model_type, filepath, checkpoint_path, checkpoint_pedal)

    audio_data = midi_obj.fluidsynth(fs=16000,
        sf2_path=f"{soundfont_path}")
    
    sr = 16000 # sample rate
    return (sr, audio_data), midi_obj

def get_piano_roll(midi_obj: pretty_midi.PrettyMIDI, fs: float) -> gr.Image:
    """
        Convert the MIDI object to a piano roll image.

        Args:
            midi_obj (pretty_midi.PrettyMIDI): MIDI object to convert.
            fs (int): Sample rate for the piano roll. Frame rate for the model
        
        Returns:
            np.ndarray: Piano roll image.
    """
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Snap2Midi Gradio app.")
    parser.add_argument("--soundfont_path", type=str, required=True, help="Path to the soundfont file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--checkpoint_pedal", type=str, required=False, default=None, \
                        help="Path to the pedal model checkpoint file (if applicable)")
    args = parser.parse_args()

    # Set soundfont path
    soundfont_path = args.soundfont_path
    checkpoint_path = args.checkpoint_path
    checkpoint_pedal = args.checkpoint_pedal

    with gr.Blocks(theme="gstaff/sketch") as demo:
        # gr.Markdown(
        #     """
        #     # Snap2Midi Gradio App
        #     """
        # )
        gr.HTML("<h1 style='margin-bottom:0;'>Snap2Midi Gradio App</h1>")

        with gr.Row(equal_height=True):
            model_type = gr.Dropdown(
                choices=[
                    "Kong",
                    "OnsetsAndFrames",
                    "HFT"
                ],
                label="Choose your AI Music Transcription Model",

            )        
            input_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Input Audio",
                autoplay=True
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

        # Set click events to make transcription possible on clicking btn and making btn invisible
        # after that
        transcribe_btn.click(process_audio, inputs=[input_audio, gr.State(soundfont_path), model_type, \
                gr.State(checkpoint_path), gr.State(checkpoint_pedal)], outputs=[out, midi_state])
        transcribe_btn.click(update_btn_visibility, inputs=[], outputs=[transcribe_btn])

        # handler events for generating a MIDI piano roll on transcription
        if model_type.value == "Kong" or model_type.value == "HFT":
            fs = 100 # frame rate for the model
        elif model_type.value == "OnsetsAndFrames":
            fs = 31.25
        else:
            raise ValueError(f"Unknown model type: {model_type.value}")
        
        out.change(get_piano_roll, inputs=[midi_state, gr.State(fs)], outputs=[piano_roll_img])
        
        # Save MIDI file and download it
        midi_filename_value = gr.State()
        midi_filename.change(lambda x: x if x else "transcription", inputs=[midi_filename],\
                              outputs=[midi_filename_value])
        download_midi_btn.click(save_midi, inputs=[midi_state, midi_filename_value], outputs=None)

    demo.launch(debug=True)
