from models.gpt_model import GPTModel
from models.claude_model import ClaudeModel
from models.deepseek_model import DeepseekModel

class ChatModelRouter:
    def __init__(self, api_keys: dict, system_message: str):
        self.models = {
            "GPT": GPTModel(api_keys.get("openai"), system_message),
            "Claude": ClaudeModel(api_keys.get("anthropic"), system_message),
            "Deepseek": DeepseekModel(api_keys.get("deepseek"), system_message),
        }

    def chat(self, prompt, history, model, stream=False):
        if model not in self.models:
            yield history + [{"role": "assistant", "content": f"Unknown model '{model}' selected."}]
            return

        if stream:
            yield from self.models[model].response_stream(prompt, history or [], stream)
        else:
            answer = self.models[model].response_full(prompt, history or [], stream)
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
