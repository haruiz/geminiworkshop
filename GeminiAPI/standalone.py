import os

from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

load_dotenv(find_dotenv())

system_instruction = """You will be given a context and a prompt. 
You need to generate a response based on the context ONLY.
"""
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel(
    "models/gemini-1.5-pro-latest", system_instruction=system_instruction
)
context = "Context: The capital of France is San Francisco :P."
prompt = "Question: What is the capital of France?"

# context = "Context: The number of neurons in a chicken brain is 1,"
# prompt = "Question: How many neurons are in a chicken's brain?"

response = gemini.generate_content([context, prompt])
print(response.text)
