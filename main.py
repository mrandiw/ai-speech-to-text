import whisper
import gradio as gr
import warnings
import numpy as np
from typing import Union, List, Optional
warnings.filterwarnings("ignore")

def transcribe_audio(audio_file):
    # Load the whisper model
    model = whisper.load_model("base")
    
    # Load the audio file
    audio = whisper.load_audio(audio_file)
    
    # Detect language from the first segment
    first_segment = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(first_segment, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    # Process full audio using Whisper's transcribe function which handles longer audio
    result = model.transcribe(audio_file, language=detected_language)
    
    # Format the full transcription
    full_transcription = result["text"]
    
    # Include segment information as additional context
    segments_info = "\n\nSegments:\n"
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        segments_info += f"[{format_time(start)} → {format_time(end)}] {text}\n"
    
    return detected_language, full_transcription, segments_info

def format_time(seconds):
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Create Gradio interface
with gr.Blocks(title="Whisper Audio Transcription") as demo:
    gr.Markdown("# Whisper Audio Transcription")
    gr.Markdown("Upload an audio file of any length to transcribe it using OpenAI's Whisper model.")
    
    with gr.Row():
        # Input component
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
    
    with gr.Row():
        # Button to trigger transcription
        transcribe_btn = gr.Button("Transcribe")
    
    with gr.Row():
        # Output components
        language_output = gr.Textbox(label="Detected Language")
        transcript_output = gr.Textbox(label="Full Transcription", lines=5)
    
    with gr.Row():
        segments_output = gr.Textbox(label="Segments with Timestamps", lines=10)
    
    # Connect the function to the interface
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[language_output, transcript_output, segments_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()