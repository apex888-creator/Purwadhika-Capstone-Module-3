import os
from dotenv import load_dotenv
from langfuse import Langfuse

# Inisialisasi klien Langfuse (mengambil API key dari environment variables)
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)


def get_prompt_from_langfuse(prompt_name: str, version: int = None):
    """Fungsi pembantu untuk mengambil prompt dari Langfuse"""
    langfuse_prompt = langfuse.get_prompt(prompt_name, version=version)
    return langfuse_prompt.compile()
