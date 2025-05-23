
from audio.audio_input import audio_receiver

def setup_voice_input(mic_input, chatbot, model_selector, audio_output_checkbox, audio_input_checkbox, router):
    def voice_message_handler(audio_data, history, model, audio_output_enabled, mic_enabled):
        
        if not audio_data or not mic_enabled:
            return history, None

        sample_rate, waveform = audio_data 
        transcript = audio_receiver(waveform,sample_rate)

        final_history = []
        for new_history in router.chat(transcript, history or [], model, audio_output_enabled):
            final_history = new_history

        return final_history, None  # clear mic UI

    mic_input.change(
        fn=voice_message_handler,
        inputs=[mic_input, chatbot, model_selector, audio_output_checkbox, audio_input_checkbox],
        outputs=[chatbot, mic_input]
    )
