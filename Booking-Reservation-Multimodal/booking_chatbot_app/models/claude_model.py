import anthropic
from audio import talker
from openai import OpenAI
import os

class ClaudeModel:
    def __init__(self, api_key, system_message):
        self.system_message = system_message
        self.client = anthropic.Anthropic(api_key=api_key)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) ## Only used for audio in talker function
    
    def message_state(self,prompt,history):
        if not prompt.strip() and history and history[-1]["role"] == "user":
            prompt = history[-1]["content"]
        return prompt.strip()

    def completion(self, prompt, stream):
        method_name = "stream" if stream else "create"
        method = getattr(self.client.messages, method_name)

        return method(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}],
        )


    def response_stream(self, prompt, history, stream):
        prompt = self.message_state(prompt, history)
        if not prompt:
            yield history + [{"role": "assistant", "content": "(Empty prompt)"}]
            return

        result = self.completion(prompt, stream)

        assistant_msg = ""

        # Only append user prompt if it's not already in history
        if not history or history[-1]["role"] != "user" or history[-1]["content"] != prompt:
            history = history + [{"role": "user", "content": prompt}]

        with result as stream:
            for text in stream.text_stream:
                assistant_msg += text or ""
                yield history + [{"role": "assistant", "content": assistant_msg}]


    def response_full(self, prompt, history, stream):
        prompt = self.message_state(prompt,history)
        if not prompt.strip():
            fallback = "(Empty prompt)"                
            return fallback  
        
        result = self.completion(prompt,stream)
        answer = result.content[0].text
        
        talker.talker_output(answer, self.openai_client)
        return answer