import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)


def get_prompt_from_langfuse(prompt_name, label="production"):
    """Fungsi pembantu untuk mengambil prompt dari Langfuse"""
    langfuse_prompt = langfuse.get_prompt(prompt_name, label=label)

    prompt_data = langfuse_prompt.prompt
    if isinstance(prompt_data, list):
        return prompt_data[0] if isinstance(prompt_data[0], str) else prompt_data[0].get("text", str(prompt_data[0]))
    return prompt_data
