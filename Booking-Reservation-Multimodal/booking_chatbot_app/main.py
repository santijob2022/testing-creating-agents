
### Importing local modules
from models.chat_models_router import ChatModelRouter
from chat_agent_system.system_messages import system_message
from ui.ui_gradio import build_gradio_ui

import os
from dotenv import load_dotenv

# ### Loading API's
load_dotenv(override=True)

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}


# ### Gradio Interface
router = ChatModelRouter(api_keys,system_message)
ui = build_gradio_ui(router)

if __name__ == "__main__":
    # ui.launch(inbrowser=True)
    ui.launch()
