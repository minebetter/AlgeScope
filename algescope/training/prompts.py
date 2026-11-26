"""One explicit base-model text protocol shared by VERL and offline evaluation."""
BASE_TEMPLATE = "{{ bos_token or '' }}{% for message in messages %}{{ message['role']|capitalize + ':\\n' + message['content'] + '\\n\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:\\n' }}{% endif %}"


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, chat_template=BASE_TEMPLATE,
                                         add_generation_prompt=True, tokenize=True)
