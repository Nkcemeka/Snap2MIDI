import gradio as gr
from .gradio_app.app import process_audio, get_piano_roll, update_btn_visibility, save_midi

def launcher(soundfont_path: str, checkpoint_path: str, checkpoint_pedal: str|None=None):
    """
        Launch the Gradio app for Snap2Midi.

        Parameters
        -----------
            soundfont_path (str): 
                Path to the soundfont file.
            checkpoint_path (str): 
                Path to the model checkpoint file.
            checkpoint_pedal (str|None, optional): 
                Path to the pedal model checkpoint file (if applicable). 
                Defaults to None.
        
        Returns
        -------
            None
    """
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
