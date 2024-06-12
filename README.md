
# Gemini Workshop

![Gemini Logo](https://raw.githubusercontent.com/haruiz/geminiplayground/main/images/logo.png)

This repository contains the code for a workshop on Generative AI using Gemini.

## Setup

1. **Install Python:** Ensure you have Python 3.10 or later installed.
2. **Install Dependencies:** Install the required packages using pip, or poetry. You can find the required packages in the `pyproject.toml` file.
3. **Set up Vertex AI and Google Cloud Credentials:**
   - Create a new Google Cloud Project following the [instructions](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
   - Enable the Vertex AI API.
   - Create a service account with the necessary permissions.
   - Download the service account key file.
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the key file.
4. **Set up AI-Studio API Key:**
   - Obtain a Google Cloud API key from the [Google AI-Studio](https://aistudio.google.com/app/apikey).
   - Set the `GOOGLE_API_KEY` environment variable to the API key.

## Code Structure

The repository is organized as follows:

- **utils.py:** Contains utility functions for the demos and examples. Such as:
    - Downloading files.
    - Normalizing URLs.
    - Getting images from URLs or paths.
    - Getting image bytes in base64 encoding.
    - Plotting embeddings.
- **ML-Foundations:**
    - This folder contains the code for the ML Foundations section of the workshop.
- **Embeddings and VAE demos:**
    - **embeddings.py:** Generates embeddings for images and their captions.
    - **vae.py:** Implements a Variational Autoencoder (VAE) for dimensionality reduction.
    - **embeddings_vertexai.py:** Client for getting multimodal embeddings using Vertex AI.
    - **embeddings_image_bind.py:** Client for getting multimodal embeddings using ImageBind.
- **GeminiAPI usage Examples:**
    - **reasoning_engine.py:** Demonstrates using Gemini as a reasoning engine with Vertex AI.
    - **standalone.py:** Simple example of using Gemini standalone for text generation.
    - **getting_started.py:** Basic introduction to using Gemini.
    - **function_calling.py:** Examples of using function calling with Gemini.
    - **context.py:** Demonstrates using Gemini with memory and retrieval-augmented generation (RAG).
    - **callbacks.py:** Shows how to use callbacks with Gemini.
    - **multimodality.py:** Examples of using Gemini with multimodal inputs (audio, image, video).
- **Applications Examples:**
    - **LLM_app.py:** Creates a simple chat app using Gemini and Vertex AI for image generation.
    - **RAG_app.py:** Creates a RAG app using Gemini and Vertex AI for document retrieval and question answering.

