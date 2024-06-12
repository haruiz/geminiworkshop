import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from tenacity import retry, stop_after_attempt
from tqdm import tqdm

from utils import generate_batches, get_image_bytes_base64

API_IMAGES_PER_SECOND = 2


class MultiModalEmbeddingsVertexAI:
    """
    Get Vertex AI embeddings client
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
    ):
        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        self.location = location
        self.project = project

    @retry(reraise=True, stop=stop_after_attempt(3))
    def get_embeddings(
        self, content: typing.Union[str, Path], content_type: str
    ) -> np.array:
        """
        Get multimodal embedding from image or text
        """
        instance = struct_pb2.Struct()
        if content_type == "text":
            instance.fields["text"].string_value = content
        elif content_type == "image":
            encoded_content = get_image_bytes_base64(content)
            image_struct = instance.fields["image"].struct_value
            image_struct.fields["bytesBase64Encoded"].string_value = encoded_content
        else:
            raise ValueError("Invalid content type")

        instances = [instance]
        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )
        response = self.client.predict(endpoint=endpoint, instances=instances)
        predictions = response.predictions[0]
        if content_type == "text":
            embedding = predictions["textEmbedding"]
        else:
            embedding = predictions["imageEmbedding"]
        return np.asarray(embedding)

    def get_embeddings_batch(
        self,
        instances: typing.Sequence[typing.Tuple[typing.Union[str, Path], str]],
        batch_size: int = 2,
    ) -> pd.DataFrame:
        """
        Get multimodal embeddings from multiple images or texts
        """
        batches = generate_batches(instances, batch_size=batch_size)
        seconds_per_job = batch_size / API_IMAGES_PER_SECOND

        def process_batch(batch):
            """
            Get embeddings for a batch
            :param batch:
            :return:
            """
            embeddings_df = []
            for content, content_type in batch:
                embeddings_arr = self.get_embeddings(content, content_type)
                embeddings_df.append(
                    pd.DataFrame(
                        [
                            {
                                **{
                                    "content": content,
                                    "content_type": content_type,
                                },
                                **{
                                    f"dim_{i}": embeddings_arr[i]
                                    for i in range(len(embeddings_arr))
                                },
                            }
                        ]
                    )
                )

            return pd.concat(embeddings_df, ignore_index=True, axis=0)

        futures = []
        with ThreadPoolExecutor() as executor:
            for batch in tqdm(batches, total=len(instances) // batch_size, position=0):
                futures.append(executor.submit(process_batch, batch))
                sleep(seconds_per_job)

        embeddings_dfs = []
        for future in futures:
            embeddings_dfs.append(future.result())
        return pd.concat(embeddings_dfs, ignore_index=True)
