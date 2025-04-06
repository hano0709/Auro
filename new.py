import gradio as gr
import numpy as np
from transformers import pipeline

# Load Whisper model
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(audio):
    try:
        if audio is None:
            return None, "⚠️ Please record something first."

        sample_rate, waveform = audio

        waveform = np.array(waveform).astype(np.float32)

        # Normalize if values are too large
        if np.max(np.abs(waveform)) > 1.0:
            waveform = waveform / 32768.0

        # If stereo, convert to mono
        if waveform.ndim > 1:
            waveform = waveform[:, 0]

        print(f"[DEBUG] waveform shape: {waveform.shape}, dtype: {waveform.dtype}, sample rate: {sample_rate}")

        result = asr_pipeline({"array": waveform, "sampling_rate": sample_rate})

        return (sample_rate, waveform), result["text"]  # ✅ swapped order

    except Exception as e:
        return None, f"❌ Transcription error: {e}"


# Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Microphone(label="🎤 Record your voice", type="numpy"),
    outputs=[
        gr.Audio(label="🔊 Playback your recording"),
        gr.Textbox(label="📝 Transcription")
    ],
    title="🎙️ Whisper Speech-to-Text",
    description="Record your voice and get real-time transcription using OpenAI Whisper."
)

interface.launch()
