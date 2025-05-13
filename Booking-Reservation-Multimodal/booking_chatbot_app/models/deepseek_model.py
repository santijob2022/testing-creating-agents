from openai import OpenAI
from audio import talker

class DeepseekModel:
    def __init__(self, api_key, system_message):
        self.system_message = system_message
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def respond_stream(self, prompt, history):
        messages = ([{"role": "system", "content": self.system_message}] +
                    history + [{"role": "user", "content": prompt}])

        assistant_msg = ""
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]

    def respond_full(self, prompt, history):
        messages = ([{"role": "system", "content": self.system_message}] +
                    history + [{"role": "user", "content": prompt}])
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        answer = response.choices[0].message.content
        talker.talker_output(answer, self.client)
        return answer
