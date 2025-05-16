import gradio as gr
from images.images import welcome_image
from ui.gradio_audio_input import setup_voice_input
from ui.gradio_text_input import setup_text_input

def build_gradio_ui(router):
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(value=[], height=500, type="messages")
            gr.Image(value=welcome_image(), height=500)

        with gr.Row():
            entry = gr.Textbox(label="Chat with our AI Assistant:")

        with gr.Row():
            audio_input_checkbox = gr.Checkbox(label="Use Microphone Input", value=False)
            audio_output_checkbox = gr.Checkbox(label="Use Voice Output", value=False)

        with gr.Row(visible=False) as audio_row:
            mic_input = gr.Audio(sources="microphone", type="filepath", label="Record your message")

        with gr.Row():
            model_selector = gr.Dropdown(["GPT", "Claude", "Deepseek"], label="Select model", value="GPT")

        with gr.Row():
            clear = gr.Button("Clear")

        def toggle_mic(checked):
            return gr.update(visible=checked)

        audio_input_checkbox.change(toggle_mic, inputs=audio_input_checkbox, outputs=audio_row)

        # Always bind both, but guard execution
        setup_voice_input(mic_input, chatbot, model_selector, audio_output_checkbox, audio_input_checkbox, router)
        setup_text_input(entry, chatbot, model_selector, audio_output_checkbox, audio_input_checkbox, router)

        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    return ui
