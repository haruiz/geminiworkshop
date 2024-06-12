import base64
import math
import os
import typing
import urllib.request
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

import cv2
import requests
import umap
import validators
from PIL import Image as PILImage
from PIL.Image import Image as PILImageType
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.cm
import matplotlib


def extract_video_frames(
    video_path: typing.Union[str, Path], video_frames_folder: typing.Union[str, Path]
) -> dict:
    """
    Extract frames from the video
    :return:
    """
    video_frames_folder = Path(video_frames_folder)
    video_frames_folder.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    vidcap = cv2.VideoCapture(str(video_path))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    duration = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    video_file_name = video_path.stem

    frame_count = 0  # Initialize a frame counter
    count = 0
    frames = {}
    with tqdm(total=math.ceil(duration), unit="sec", desc="Extracting frames") as pbar:
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break
            if count % int(fps) == 0:  # Extract a frame every second

                frame_count += 1
                file_name_prefix = os.path.basename(video_file_name).replace(".", "_")
                frame_prefix = f"_frame"
                frame_image_filename = (
                    f"{file_name_prefix}{frame_prefix}{frame_count:04d}.jpg"
                )
                minutes = frame_count // 60
                seconds = frame_count % 60
                time_string = f"{minutes:02d}:{seconds:02d}"
                frame_image_path = video_frames_folder.joinpath(frame_image_filename)
                frames[time_string] = frame_image_path
                cv2.imwrite(str(frame_image_path), frame)
                pbar.update(1)
            count += 1
    vidcap.release()
    return frames


def pretty_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def download_file(file_uri: str, file_path: typing.Union, chunk_size=1024):
    """
    This function downloads a file from a given URL to a specified file path.
    :param file_uri: The URL of the file to download.
    :param file_path: The file path to save the downloaded file.
    :param chunk_size: The size of the chunk to download.
    :return:
    """
    file_path = Path(file_path)
    if file_path.exists():
        print(f"File {file_path} already exists, skipping download.")
        return
    r = requests.get(file_uri, stream=True)
    total_length = int(r.headers.get("content-length"))
    with open(file_path, "wb") as f, alive_bar(
        manual=True, title="Downloading File...."
    ) as bar:
        downloaded_bytes = 0
        for data in r.iter_content(chunk_size):
            downloaded_bytes += len(data)
            msg = f"{pretty_size(downloaded_bytes)} - {pretty_size(total_length)}"
            bar(downloaded_bytes / total_length)
            bar.text(msg)
            f.write(data)


def seconds_to_time_string(seconds):
    """Converts an integer number of seconds to a string in the format '00:00'.
    Format is the expected format for Gemini 1.5.
    """
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def get_timestamp_seconds(filename, prefix):
    """Extracts the frame count (as an integer) from a filename with the format
    'output_file_prefix_frame0000.jpg'.
    """
    parts = filename.split(prefix)
    if len(parts) != 2:
        return None  # Indicate that the filename might be incorrectly formatted

    frame_count_str = parts[1].split(".")[0]
    return int(frame_count_str)


def get_output_file_prefix(filename, prefix):
    """Extracts the output file prefix from a filename with the format
    'output_file_prefix_frame0000.jpg'.
    """
    parts = filename.split(prefix)
    if len(parts) != 2:
        return None  # Indicate that the filename might be incorrectly formatted

    return parts[0]


def get_frame_timestamp(filename):
    """Extracts the frame timestamp from a filename with the format
    'output_file_prefix_frame0000.jpg'.
    """
    seconds = get_timestamp_seconds(str(filename), "frame")
    return seconds_to_time_string(seconds)


def generate_batches(
    inputs: typing.Sequence[typing.Tuple[typing.Union[str, Path], str]], batch_size: int
) -> typing.Generator[typing.List[str], None, None]:
    """
    Generator function that takes a list of strings and a batch size, and yields batches of the specified size.
    """

    for i in range(0, len(inputs), batch_size):
        yield inputs[i : i + batch_size]


def normalize_url(url: str) -> str:
    """
    converts gcs uri to url for image display.
    """
    url_parts = urlparse(url)
    scheme = url_parts.scheme
    if scheme == "gs":
        return "https://storage.googleapis.com/" + url.replace("gs://", "").replace(
            " ", "%20"
        )
    elif scheme in ["http", "https"]:
        return url
    raise Exception("Invalid scheme")


def get_image_from_url(url: str) -> PILImageType:
    """
    Create an image from url and return it
    """
    http_uri = normalize_url(url)
    try:
        assert validators.url(http_uri), "invalid url"
        resp = urllib.request.urlopen(url, timeout=30)
        image = PILImage.open(resp)
        return image
    except HTTPError as err:
        if err.strerror == 404:
            raise Exception("Image not found")
        elif err.code in [403, 406]:
            raise Exception("Forbidden image, it can not be reached")
        else:
            raise


def get_image_from_path(path: str) -> PILImageType:
    """
    Read image from file and return it
    """
    return PILImage.open(path)


def get_image_from_anywhere(uri_or_path: typing.Union[str, Path]) -> PILImageType:
    """
    read an image from an url or local file and return it
    """
    uri_or_path = str(uri_or_path)
    if validators.url(uri_or_path):
        return get_image_from_url(uri_or_path)
    else:
        return get_image_from_path(uri_or_path)


def get_image_bytes_base64(image_path: typing.Union[str, Path]) -> str:
    """
    Get image bytes from image path
    """
    image = get_image_from_anywhere(image_path)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plot_embeddings(
    embeddings_arr,
    embeddings_labels,
    method="tsne",
    show_labels=True,
    show_legends=False,
    ax=None,
    **kwargs,
):
    """
    Plots embeddings using dimensionality reduction (t-SNE, PCA, or UMAP) and returns the matplotlib axes.

    :param embeddings_arr: Array of embeddings (high-dimensional).
    :param embeddings_labels: Labels for each point.
    :param method: Method for dimensionality reduction ('tsne', 'pca', or 'umap').
    :param show_labels: Boolean to show labels next to points.
    :param kwargs: Additional keyword arguments for the reduction method.
    :return: Matplotlib axes containing the plot.
    """
    # Choose the reduction method
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=0, **kwargs)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, **kwargs)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, **kwargs)
    else:
        raise ValueError("Unsupported dimensionality reduction method")

    # Apply dimensionality reduction
    projected_points = reducer.fit_transform(embeddings_arr)

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots()
    cmap = matplotlib.colormaps.get_cmap("tab10")
    cmap = matplotlib.colors.ListedColormap(cmap.colors)
    colors = [cmap(i) for i in range(len(set(embeddings_labels)))]

    # Plot points
    for i, label in enumerate(set(embeddings_labels)):
        indices = [j for j, x in enumerate(embeddings_labels) if x == label]
        x, y = projected_points[indices, 0], projected_points[indices, 1]
        ax.scatter(x, y, color=colors[i], label=label)
        if show_labels:
            for xi, yi in zip(x, y):
                ax.text(xi, yi, label)

    if show_legends:
        ax.legend()
    return ax  # Return the axes for further manipulation
