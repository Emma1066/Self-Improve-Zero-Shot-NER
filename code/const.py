
model_list = {
    "gpt-3.5-turbo": {"abbr":"tb", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-16k": {"abbr":"tb", "publisher":"openai", "max_tokens":4096*4},
    "gpt-3.5-turbo-0613": {"abbr":"tb0613", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-instruct": {"abbr":"tbinstruct", "publisher":"openai", "max_tokens":4096},
    "gpt-3.5-turbo-1106": {"abbr":"tb1106", "publisher":"openai", "max_tokens":4096*4},
    "text-davinci-003": {"abbr":"d3", "publisher":"openai", "max_tokens":4097},
}

dataset_language_map = {
    "wikigold": "en",
}

my_openai_api_keys = [
    {"key":"YOUR_KEY", "set_base":True, "api_base":"YOUR_API_BASE"},
]
