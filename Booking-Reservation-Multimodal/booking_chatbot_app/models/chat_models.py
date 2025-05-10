# chat_models.py

from openai import OpenAI
import anthropic
# import os
from audio import talker

class ChatModelRouter:
    def __init__(self, api_keys: dict, system_message: str):
        self.api_keys = api_keys
        self.system_message = system_message
        self.models = {}

    def _get_openai_client(self, base_url=None, api_key=None):
        return OpenAI(
            api_key=api_key or self.api_keys.get("openai"),
            base_url=base_url
        )

    def chat(self, prompt, history, model):
        dispatch = {
            "GPT": self.chat_with_gpt,
            "Claude": self.chat_with_claude,
            "Deepseek": self.chat_with_deepseek,
        }
        if model not in dispatch:
            yield history + [{"role": "assistant", "content": f"Unknown model '{model}' selected."}]
            return

        yield from dispatch[model](prompt, history)

    def chat_with_gpt(self, prompt, history):
        if history is None:
            history = []

        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": prompt}]
        openai_client = self._get_openai_client(api_key=self.api_keys.get("openai"))

        assistant_msg = ""
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]
        
        self.talker_output(assistant_msg)

    def chat_with_claude(self, prompt, history):
        if history is None:
            history = []

        if not prompt.strip() and history and history[-1]["role"] == "user":
            prompt = history[-1]["content"]
        if not prompt.strip():
            yield history + [{"role": "assistant", "content": "(Empty prompt)"}]
            return

        claude_client = anthropic.Anthropic(api_key=self.api_keys.get("anthropic"))        
        result = claude_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_msg = ""
        with result as stream:
            for text in stream.text_stream:
                assistant_msg += text or ""
                yield history + [{"role": "assistant", "content": assistant_msg}]

        self.talker_output(assistant_msg)

    def chat_with_deepseek(self, prompt, history):
        if history is None:
            history = []

        messages = [{"role": "system", "content": self.system_message}] + history + [{"role": "user", "content": prompt}]
        deepseek_client = self._get_openai_client(
            api_key=self.api_keys.get("deepseek"),
            base_url="https://api.deepseek.com"
        )

        assistant_msg = ""
        stream = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]
                ## In case the talker should be activated
        self.talker_output(assistant_msg)
    
    def talker_output(self,assistant_msg):
        openai_client = self._get_openai_client(api_key=self.api_keys.get("openai"))
        talker.talker(assistant_msg, openai_client)
