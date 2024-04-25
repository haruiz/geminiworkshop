import os

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

load_dotenv(find_dotenv())

console = Console()

if __name__ == "__main__":
    config = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = genai.list_models()
    for model in models:
        console.print(model)

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    prompt = "Why is the sky blue?"
    num_prompt_tokens = model.count_tokens(prompt)
    console.print(f"👨‍🚀: {prompt}, number of tokens: {num_prompt_tokens}")
    responses = model.generate_content(prompt, stream=True)
    for response in responses:
        console.print(response.text, end="")
