
### Importing local modules
from audio import talker
from images.images import welcome_image
from models.chat_models_router import ChatModelRouter

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr


# ### Loading API's
load_dotenv(override=True)

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}

if api_keys.get("openai"):
    print(f"OpenAI API Key exists")
else:
    print("OpenAI API Key not set")
    
if api_keys.get("anthropic"):
    print(f"Anthropic API Key exists")
else:
    print("Anthropic API Key not set")

if api_keys.get("deepseek"):
    print(f"Deepseek API Key exists")
else:
    print("Deepseek API Key not set")


# ### Making the connections
openai = OpenAI()


# ### Defining the chatbot system

# Modify this according to system updates
system_message = "You are a helpful assistant for making booking reservations. "
system_message += "Give short, courteous answers, no more than 1 sentence."
system_message += "Always be accurate. If you don't know the answer, say so."


router = ChatModelRouter(api_keys,system_message)
# ### Gradio Interface

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(value=[],height=500, type="messages")
        image_output = gr.Image(value=welcome_image(openai),height=500)
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

if __name__ == "__main__":
    ui.launch(inbrowser=True)
