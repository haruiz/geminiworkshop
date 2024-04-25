from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv(find_dotenv())

gemini = GoogleGenerativeAI(model="gemini-1.5-pro-latest")
response = gemini.invoke("What is the capital of France?")
print(response)
