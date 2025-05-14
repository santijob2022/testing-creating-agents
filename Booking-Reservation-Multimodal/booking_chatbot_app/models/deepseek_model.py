from openai import OpenAI
from audio import talker
import os

class DeepseekModel:
    def __init__(self, api_key, system_message):
        self.system_message = system_message
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) ## Only used for audio in talker function
    
    def messages_state(self,prompt, history):
        messages = ([{"role": "system", "content": self.system_message}] +
            history + [{"role": "user", "content": prompt}])  
        return messages  

    def completion(self, messages,stream):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=stream
        )
        return response

    def response_stream(self, prompt, history,stream):
        messages = self.messages_state(prompt,history)        
        stream = self.completion(messages, stream)

        assistant_msg = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            assistant_msg += delta
            yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]

    def response_full(self, prompt, history,stream):
        messages = self.messages_state(prompt,history)
        response = self.completion(messages, stream)

        answer = response.choices[0].message.content

        talker.talker_output(answer, self.openai_client)

        return answer
