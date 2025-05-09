#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

### Importing local modules
from audio import talker

import warnings
import logging

# Suppress user warnings
warnings.filterwarnings("ignore")

# Suppress Gradio & uvicorn logs
logging.getLogger("gradio").setLevel(logging.ERROR)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr


# ### Loading API's

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if anthropic_api_key:
    print(f"Deepseek API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Deepseek API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# ### Making the connections

openai = OpenAI()

claude = anthropic.Anthropic()

deepseek = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)


# ### Creating Welcome Image with dall-e-3


import base64
from io import BytesIO
from PIL import Image

def welcome_image():
    """Function to generate the welcome image in the chatbot making reference to the Booking experience."""

    if os.path.exists("welcome_image_booking.png"):
        image = Image.open("welcome_image_booking.png")
    else:
        image_response = openai.images.generate(
                model="dall-e-3",
                prompt=f"""Create a photorealistic welcome image with the text 
                "Welcome!"
                in a clear, elegant, and formal font.             
                In the center, there's a modern glass hut. 
                In front of the hut, a couple is peacefully sitting on the ground, gently illuminated by sun rays 
                filtering through the canopy. Around the border of the image, the subtle silhouettes of curious 
                jungle animals are partially visible, watching the couple with interest. 
                The overall mood is serene and inviting.
                The background features a lush jungle with vivid greenery.
                """,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
                style="vivid"       
            )
        image_base64 = image_response.data[0].b64_json
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Save the image
        image.save("welcome_image_booking.png", format="PNG")
        print("Image generated and saved.")
    # Display the image
    # display(image)
    
    return image



# ### Defining the chatbot system

# Modify this according to system updates
system_message = "You are a helpful assistant for making booking reservations. "
system_message += "Give short, courteous answers, no more than 1 sentence."
system_message += "Always be accurate. If you don't know the answer, say so."


# ### Defining the selected model in streaming mode
# 
# This depends on the model chosen by the user and will also be useful to pick a model to answer certain type of questions

def chat(prompt, history, model):
    if history is None:
        history = []

    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": prompt}]

    # print("History is:")
    # print(history)
    # print("And messages is:")
    # print(messages)

    assistant_msg = ""

    if model == "GPT":
        stream = openai.chat.completions.create(
            model="gpt-4o-mini",  # or use another GPT variant
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]

    elif model == "Claude":
        # Try to recover original user prompt from last history item
        if not prompt.strip() and history and history[-1]["role"] == "user":
            prompt = history[-1]["content"]

        if not prompt.strip():
            yield history + [{"role": "assistant", "content": "(Empty prompt)"}]
            return

        system = system_message.strip() if system_message and system_message.strip() else None
        result = claude.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_msg = ""
        with result as stream:
            for text in stream.text_stream:
                assistant_msg += text or ""
                yield history + [{"role": "assistant", "content": assistant_msg}]


    elif model == "Deepseek":
        stream = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]

    else:
        # fallback or unknown model
        assistant_msg = "Unknown model selected."
        yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]
    
    talker.talker(assistant_msg, openai)


# ### Gradio Interface

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
                     chat, 
                     inputs=[entry, chatbot, model_selector], 
                     outputs=chatbot
                     ).then(
                         lambda: "", None, entry
                     )

    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

# ui.launch(inbrowser=True, server_port=7868,prevent_thread_lock=True)

if __name__ == "__main__":
    ui.launch(inbrowser=True)
