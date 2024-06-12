import warnings

from matplotlib import pyplot as plt

from utils import get_image_bytes_base64, plot_embeddings
from sklearn.metrics.pairwise import cosine_similarity

import typing
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
import seaborn as sns

# from embeddings_image_bind import MultimodalEmbeddingsImageBind
from embeddings_vertexai import MultiModalEmbeddingsVertexAI

warnings.filterwarnings("ignore")
console = Console()

load_dotenv(find_dotenv())


def get_image_caption(
    generative_model: ChatGoogleGenerativeAI, image_path: typing.Union[str, Path]
) -> str:
    """
    Get image caption from image path
    """
    image_base64 = get_image_bytes_base64(image_path)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "generate a label for the following image using max two words:",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )
    response = generative_model.invoke([message])
    return response.content


def generate_embeddings():
    """
    Generate embeddings for images and their captions found in the data/images directory
    """
    images = Path("data/images").rglob("*.jpg")
    image_embedding_client = MultiModalEmbeddingsVertexAI(
        project="build-with-ai-project"
    )
    # image_embedding_client = MultimodalEmbeddingsImageBind()
    gemini = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    multimodal_instances = []
    for image in images:
        multimodal_instances.append((image, "image"))
        multimodal_instances.append((get_image_caption(gemini, image), "text"))
    embeddings_df = image_embedding_client.get_embeddings_batch(
        multimodal_instances, batch_size=2
    )
    embeddings_df.to_csv("embeddings.csv", index=False)
    return embeddings_df


# def cosine_similarity(a: np.array, b: np.array) -> float:
#     """
#     Calculate cosine similarity between two vectors
#     """
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_a * norm_b)


if __name__ == "__main__":
    # Assuming embeddings_df is already generated
    embeddings_df = generate_embeddings()
    embeddings_arr = np.asarray(embeddings_df.iloc[:, 2:])
    embeddings_labels = embeddings_df["content"]
    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_arr)
    # Create a DataFrame from the similarity matrix
    similarity_df = pd.DataFrame(
        similarity_matrix, columns=embeddings_labels, index=embeddings_labels
    )
    # Now similarity_df contains the similarity scores between all pairs
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_df, annot=True, cmap="coolwarm")
    plt.savefig("similarity_matrix.png")
    plt.show()

    reducers = {
        "pca": {},
        "tsne": {"perplexity": 5},
        "umap": {"n_neighbors": 5, "min_dist": 0.3},
    }
    fig = plt.figure(figsize=(15, 10))
    num_cols = 3
    num_rows = len(reducers) // num_cols
    for i, (method, kwargs) in enumerate(reducers.items()):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        plot_embeddings(
            embeddings_arr,
            embeddings_labels,
            method=method,
            show_labels=True,
            show_legends=False,
            ax=ax,
            **kwargs,
        )
        ax.set_title(method)
    plt.tight_layout()
    plt.savefig("embeddings.png")
    plt.show()
