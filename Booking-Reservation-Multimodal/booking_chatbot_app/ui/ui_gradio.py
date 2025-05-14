import gradio as gr
from images.images import welcome_image


def build_gradio_ui(router):

    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(value=[],height=500, type="messages")
            image_output = gr.Image(value=welcome_image(),height=500)
        with gr.Row():        
            entry = gr.Textbox(label="Chat with our AI Assistant:")
        with gr.Row():        
            model_selector = gr.Dropdown(["GPT", "Claude", "Deepseek"], label="Select model", value="GPT")
        with gr.Row():
            clear = gr.Button("Clear")        

        def do_entry(message, history, model):
            if not message.strip():
                return "", history, model  # Do nothing if prompt is empty
            if history is None:
                history = []
            history += [{"role": "user", "content": message}]
            return "", history, model

        entry.submit(fn=do_entry, 
                    inputs=[entry, chatbot, model_selector], 
                    outputs=[entry, chatbot,model_selector]).then(
                        router.chat, 
                        inputs=[entry, chatbot, model_selector], 
                        outputs=chatbot
                        ).then(
                            lambda: "", None, entry
                        )

        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    return ui