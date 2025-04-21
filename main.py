import whisper
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

def transcribe_audio(audio_file):
    # Load the whisper model
    model = whisper.load_model("base")
    
    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    
    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    
    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    
    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    
    # Return results
    return detected_language, result.text

# Create Gradio interface
with gr.Blocks(title="Whisper Audio Transcription") as demo:
    gr.Markdown("# Whisper Audio Transcription")
    gr.Markdown("Upload an audio file to transcribe it using OpenAI's Whisper model.")
    
    with gr.Row():
        # Input component
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
    
    with gr.Row():
        # Button to trigger transcription
        transcribe_btn = gr.Button("Transcribe")
    
    with gr.Row():
        # Output components
        language_output = gr.Textbox(label="Detected Language")
        transcript_output = gr.Textbox(label="Transcription", lines=5)
    
    # Connect the function to the interface
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[language_output, transcript_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()