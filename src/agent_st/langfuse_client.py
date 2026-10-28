from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()


def get_prompt_from_langfuse(prompt_name, label="production"):
    """Fungsi pembantu untuk mengambil prompt dari Langfuse"""
    langfuse_prompt = langfuse.get_prompt(prompt_name, label=label)

    prompt_data = langfuse_prompt.prompt
    if isinstance(prompt_data, list):
        return prompt_data[0] if isinstance(prompt_data[0], str) else prompt_data[0].get("text", str(prompt_data[0]))
    return prompt_data
