import os
from urllib.parse import urlparse

from torch.hub import download_url_to_file


def load_file_from_url(
    url: str,
    *,
    model_dir: str = None,
    progress: bool = True,
    file_name: str = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "layer_model")
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, progress=progress)
    return cached_file
