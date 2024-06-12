import time

import google.generativeai as genai
from diskcache import Cache
from dotenv import load_dotenv, find_dotenv
from google.generativeai.types import File
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
import validators
from pathlib import Path
import os
import typing
from utils import download_file

load_dotenv(find_dotenv())

console = Console()

cache = Cache(directory=".cache")


def upload_and_cache_file(file_path: typing.Union[str, Path]) -> File:
    """
    Uploads a file to Google Storage and caches it locally.

    If the file_path is a URL, it downloads the file first.
    Then, it checks the cache for the file; if not present, uploads the file and caches the result.

    :param file_path: Path to the file or URL of the file to upload.
    :return: The uploaded file URL or identifier.
    """
    file_name = Path(file_path).name
    file_path = str(file_path)

    # Download if file_path is a URL
    if validators.url(file_path):
        download_file(file_path, file_name)
        file_path = file_name

    # Check cache first
    uploaded_file = cache.get(file_name)
    if uploaded_file is None:
        uploaded_file = genai.upload_file(file_path)
        cache.set(file_name, uploaded_file)

    return uploaded_file


def audio_example():
    """
    This function demonstrates how to use Gemini to analyze audio data.
    :return:
    """
    audio_file_uri = "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
    uploaded_file = upload_and_cache_file(audio_file_uri)

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    num_tokens = model.count_tokens(uploaded_file)
    console.print(f"Uploaded file: {uploaded_file.name}, num tokens: {num_tokens}")
    prompt = "Listen carefully to the following audio file. Provide a brief summary."
    response_chunks = model.generate_content([prompt, uploaded_file], stream=True)
    for chunk in response_chunks:
        console.print(chunk.text, end="")


def image_example():
    """
    This function demonstrates how to use the Gemini to analyze image data.
    :return:
    """
    image_file_uri = (
        "https://m.media-amazon.com/images/I/71LMIioO5tL._AC_UF894,1000_QL80_.jpg"
    )
    uploaded_file = upload_and_cache_file(image_file_uri)

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    num_tokens = model.count_tokens(uploaded_file)
    console.print(f"Uploaded file: {uploaded_file.name}, num tokens: {num_tokens}")
    prompt = "Describe the image below in detail."
    response_chunks = model.generate_content([prompt, uploaded_file], stream=True)
    for chunk in response_chunks:
        console.print(chunk.text, end="")


def video_example(force_download=False):
    """
    This function demonstrates how to use the Gemini to analyze video data.
    :return:
    """
    # https://gist.github.com/jsturgis/3b19447b304616f18657
    video_file_uri = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    uploaded_file = upload_and_cache_file(video_file_uri)
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(10)
        uploaded_file = genai.get_file(uploaded_file.name)
    if uploaded_file.state.name == "FAILED":
        raise Exception("File upload failed")
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    prompt = ["Describe the following video:", uploaded_file]
    response_chunks = model.generate_content(prompt, stream=True)
    for chunk in response_chunks:
        console.print(chunk.text, end="")


def langchain_example():
    """
    This function demonstrates how to use langchain API to generate content based on a multimodal prompt using Gemini.
    :return:
    """
    """
    This function demonstrates how to use the GoogleGenerativeAI model to generate content based on a multimodal prompt.
    :return:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    text_part = {
        "type": "text",
        "text": "Use the following image to address the user query:",
    }
    image_part = {
        "type": "image_url",
        "image_url": "https://m.media-amazon.com/images/I/71LMIioO5tL._AC_UF894,1000_QL80_.jpg",
    }
    messages = [
        SystemMessage(
            content="you are PICASSO!, use the following image to address the user query:"
        ),
        HumanMessage(content=[text_part, image_part]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()
    ai_message = chain.invoke({"prompt": "How many animals are in the image?"})
    print(ai_message)


if __name__ == "__main__":
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    audio_example()
