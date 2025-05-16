
import numpy as np
import tempfile
import soundfile as sf
from openai import OpenAI


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1.0, 1.0] range. That's why we are dividing by the maximum."""
    if np.max(np.abs(audio)) == 0:
        return audio
    return audio / np.max(np.abs(audio))

def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove DC offset from signal: Moves the signal to be centered at 0"""
    return audio - np.mean(audio)

def audio_receiver(numpy_audio,sample_rate):
    client = OpenAI()

    print("Received numpy_audio type:", type(numpy_audio), flush=True)
    print("numpy_audio value:", numpy_audio, flush=True)
    print("sample_rate:", sample_rate, flush=True)

    if not isinstance(numpy_audio, np.ndarray) or numpy_audio.size == 0:
        raise ValueError("Invalid or empty audio input")

    # Convert to NumPy array
    audio = np.asarray(numpy_audio)

    # Clean audio
    audio = remove_dc_offset(audio)
    audio = normalize_audio(audio)

    # Ensure 2D shape
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)  # (n_samples,) → (n_samples, 1)

    # log the shape
    print("audio.shape =", audio.shape, "dtype:", audio.dtype, flush=True)

    # Write and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, samplerate=sample_rate)
        tmp_path = tmp.name  # Capture path
        # print("Temp audio written to:", tmp_path)

    # Open file again in binary mode
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    import os
    os.remove(tmp_path) # Remove temp file

    return transcript.text


