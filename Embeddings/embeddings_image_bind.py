import typing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import generate_batches

try:
    from imagebind import data
    import torch
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

except ImportError:
    raise Exception(
        "Install imagebind package to be able to use the MultimodalImageBindClient"
    )


class MultimodalEmbeddingsImageBind:
    """
    Client for getting multimodal embeddings from image or text using the ImageBind model
    ImageBind: One Embedding Space To Bind Them All
    https://github.com/facebookresearch/ImageBind

    git clone https://github.com/facebookresearch/ImageBind
    cd ImageBind
    pip install -e .
    """

    def __init__(self):
        # Instantiate model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

    def getEmbeddingVector(self, inputs) -> np.ndarray:
        """
        Get embedding vector from image or text
        :return:
        """
        with torch.no_grad():
            embedding = self.model(inputs)
        for key, value in embedding.items():
            vec = value.reshape(-1)
            vec = vec.numpy()
            return vec

    def get_embeddings(
        self, content: typing.Union[str, Path], content_type: str
    ) -> np.ndarray:
        """
        Get multimodal embedding from image or text
        """
        if content_type == "image":
            data_path = [content]
            inputs = {
                ModalityType.VISION: data.load_and_transform_vision_data(
                    data_path, self.device
                )
            }
        elif content_type == "text":
            txt = [content]
            inputs = {ModalityType.TEXT: data.load_and_transform_text(txt, self.device)}
        else:
            raise ValueError(
                "Invalid content type. Only 'image' and 'text' are supported."
            )
        vec = self.getEmbeddingVector(inputs)
        return vec

    def get_embeddings_batch(
        self,
        instances: typing.Sequence[typing.Tuple[typing.Union[str, Path], str]],
        batch_size: int = 2,
    ) -> pd.DataFrame:
        """
        Get multimodal embeddings from multiple images or texts
        """
        batches = generate_batches(instances, batch_size=batch_size)
        embeddings_df = []
        for batch in tqdm(batches, total=len(instances) // batch_size, position=0):
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
