import tiktoken


def token_count(model_id, text):
    encoding = tiktoken.encoding_for_model(model_id)
    return len(encoding.encode(text))
