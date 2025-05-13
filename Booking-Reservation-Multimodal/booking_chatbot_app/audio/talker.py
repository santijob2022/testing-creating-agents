#### Defining Audio Output
# The audio section will need to install ffmpeg to work on Windows.

from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import threading

def talker_output(message,openai):
    # Generate speech
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )
    
    # Decode MP3 audio
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")

    # Play in a background thread
    threading.Thread(target=play, args=(audio,), daemon=True).start()