
from openai import OpenAI

def audio_receiver(audio_path):
    client = OpenAI()
    
    if not audio_path:
            return "No audio input received."
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            # model="gpt-4o-transcribe",
            file=f
        )
    return transcript.text

