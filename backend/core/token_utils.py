import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")  # close approximation for most modern LLMs

def estimate_tokens(text: str) -> int:
    return len(_encoder.encode(text or ""))