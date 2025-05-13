import anthropic
from audio import talker

class ClaudeModel:
    def __init__(self, api_key, system_message):
        self.system_message = system_message
        self.client = anthropic.Anthropic(api_key=api_key)

    def respond_stream(self, prompt, history):
        if not prompt.strip():
            if history and history[-1]["role"] == "user":
                prompt = history[-1]["content"]
            if not prompt.strip():
                yield history + [{"role": "assistant", "content": "(Empty prompt)"}]
                return

        assistant_msg = ""
        result = self.client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}],
        )

        with result as stream:
            for text in stream.text_stream:
                assistant_msg += text or ""
                yield history + [{"role": "assistant", "content": assistant_msg}]

    def respond_full(self, prompt, history):
        result = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=self.system_message,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = result.content[0].text
        talker.talker_output(answer, None)
        return answer