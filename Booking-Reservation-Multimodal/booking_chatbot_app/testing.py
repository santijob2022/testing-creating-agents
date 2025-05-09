def chat_full_response(prompt, history, model):
    if history is None:
        history = []

    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": prompt}]

    assistant_msg = ""

    if model == "GPT":
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        assistant_msg = response.choices[0].message.content

    elif model == "Claude":
        system = system_message.strip() if system_message and system_message.strip() else None
        result = claude.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        assistant_msg = result.content[0].text

    elif model == "Deepseek":
        response = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        assistant_msg = response.choices[0].message.content

    else:
        assistant_msg = "Unknown model selected."

    # Call TTS to speak the final message (non-blocking if talker uses threading)
    talker(assistant_msg)

    # Return the full message at once
    yield history + [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_msg}]
