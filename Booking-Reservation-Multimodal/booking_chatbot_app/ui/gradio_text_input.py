
def setup_text_input(entry, chatbot, model_selector, audio_output_checkbox, audio_input_checkbox, router):
    def text_entry(message, history, model, mic_enabled):
        if mic_enabled:  # don't handle if mic is enabled
            return "", history, model

        if not message.strip():
            return "", history, model
        if history is None:
            history = []
        history += [{"role": "user", "content": message}]
        return "", history, model

    entry.submit(
        fn=text_entry,
        inputs=[entry, chatbot, model_selector, audio_input_checkbox],
        outputs=[entry, chatbot, model_selector]
    ).then(
        router.chat,
        inputs=[entry, chatbot, model_selector, audio_output_checkbox],
        outputs=chatbot
    ).then(lambda: "", None, entry)
