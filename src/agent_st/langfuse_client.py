from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()


def get_prompt_from_langfuse(prompt_name, label="production"):
    """Fungsi pembantu untuk mengambil prompt dari Langfuse.

    Mengembalikan teks system prompt (str), baik prompt bertipe text
    maupun chat.
    """
    langfuse_prompt = langfuse.get_prompt(prompt_name, label=label)
    prompt_data = langfuse_prompt.prompt

    if isinstance(prompt_data, list):
        for message in prompt_data:
            if message.get("role") == "system":
                return message["content"]
        raise ValueError(
            f"Prompt '{prompt_name}' tidak punya pesan dengan role 'system'.")
    return prompt_data
