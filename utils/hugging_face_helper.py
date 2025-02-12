import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Union, Optional, List

from tqdm import tqdm
from huggingface_hub import HfApi


# Constants
NOTSOFAR_HF_REPO_ID = "microsoft/NOTSOFAR"

# Initialize Hugging Face API
hugging_face_token = os.getenv('HF_TOKEN')
assert hugging_face_token, ("HuggingFace token not found. Please set the HF_TOKEN environment variable, "
                            "if you have set it, please restart the session. "
                            "Use README.md (NOTSOFAR-1 Datasets - Download Instructions) for more information.")
_HF_API = HfApi(token=hugging_face_token)


def is_hf_dir_exists(subfolder: Union[str, Path]) -> bool:
    """
    Check if a subfolder exists in the Hugging Face repository

    Args:
        subfolder: path in the repository to check

    Returns:
        bool: True if the subfolder exists, False otherwise
    """
    assert isinstance(subfolder, (str, Path)), "local_dir should be a string or Path object"

    try:
        files = _HF_API.list_repo_tree(repo_id=NOTSOFAR_HF_REPO_ID, repo_type="dataset", path_in_repo=subfolder)
        return True if files else False
    except Exception as e:
        return False


def list_hf_dir(root_dir: Union[str, Path], recursive: bool = False) -> List[str]:
    """
    List all files in a specified directory in the Hugging Face repository.

    Args:
        root_dir (Union[str, Path]): The root directory in the repository to list files from.
        recursive (bool): Whether to list files recursively. Defaults to False.

    Returns:
        List[str]: A list of file paths in the specified directory.

    Raises:
        AssertionError: If the specified directory does not exist in the repository.
    """
    root_dir = str(root_dir)
    assert is_hf_dir_exists(root_dir), f"Cannot find {root_dir} in the Hugging Face repository"

    try:
        return [val.path for val in _HF_API.list_repo_tree(
            repo_id=NOTSOFAR_HF_REPO_ID, repo_type="dataset", path_in_repo=root_dir, recursive=recursive)]
    except Exception as e:
        raise RuntimeError(f"Failed to list directory {root_dir} in the Hugging Face repository: {e}")


def list_hf_dir_files(root_dir: Union[str, Path]) -> List[str]:
    """
    List all files (excluding directories) in a specified directory recursively in the Hugging Face repository.

    Args:
        root_dir (Union[str, Path]): The root directory in the repository to list files from.

    Returns:
        List[str]: A list of file paths in the specified directory.
    """
    def _is_file(file_path: str) -> bool:
        return '.' in os.path.basename(file_path)

    try:
        return [dir_file_path for dir_file_path in list_hf_dir(root_dir, recursive=True) if _is_file(dir_file_path)]
    except Exception as e:
        raise RuntimeError(f"Failed to list files in directory {root_dir} in the Hugging Face repository: {e}")


def download_hf_file(file_path: str, local_dir: str, pbar: Optional[tqdm] = None) -> str:
    """
    Download a file from the Hugging Face repository.

    Args:
        file_path (str): Path of the file in the repository.
        local_dir (Path): Local directory to download the file to.
        pbar (Optional[tqdm]): tqdm progress bar object (optional).

    Returns:
        str: Local file path where the file is downloaded.

    Raises:
        RuntimeError: If the file download fails.
    """
    local_file_path = Path(local_dir) / file_path
    os.makedirs(local_file_path.parent, exist_ok=True)

    try:
        _HF_API.hf_hub_download(repo_id=NOTSOFAR_HF_REPO_ID, filename=file_path,
                                repo_type="dataset", local_dir=local_dir)
        if pbar:
            pbar.update(1)  # Increment progress bar if provided
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_path} from the Hugging Face repository: {e}")

    return str(local_file_path)


def download_hf_dir(subfolder: str, local_dir: Union[str, Path], max_workers: int = os.cpu_count()) -> List[str]:
    """
    Download all files in a subfolder of the Hugging Face repository.

    Args:
        subfolder (str): Path in the repository to download.
        local_dir (Union[str, Path]): Local directory to download the files to.
        max_workers (int): Number of workers to use for downloading, defaults to number of CPUs.

    Returns:
        List[str]: A list of file paths downloaded to the local directory.

    Raises:
        AssertionError: If the subfolder does not exist in the repository.
        ValueError: If the local_dir is not a string or Path object.
        ValueError: If max_workers is not an integer.
        RuntimeError: If downloading files fails.
    """
    if not is_hf_dir_exists(subfolder):
        raise AssertionError(f"Subfolder {subfolder} does not exist in the Hugging Face repository")
    if not isinstance(local_dir, (str, Path)):
        raise ValueError("local_dir should be a string or Path object")
    if not isinstance(max_workers, int):
        raise ValueError("max_workers should be an integer")

    local_dir = str(local_dir)
    files = list_hf_dir_files(root_dir=subfolder)

    with tqdm(total=len(files), desc="Downloading", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_hf_file, file, local_dir, pbar) for file in files]
            wait(futures)
    return files


def main():
    """
    Usage example for the Hugging Face helper functions
    """
    import tempfile

    # List directory
    print("\n>>> Listing directory")
    folder_path = "benchmark-datasets/dev_set"
    dirs = list_hf_dir(root_dir=folder_path)
    print(f"Directories in {folder_path}: {dirs}")

    # List files in a directory
    print("\n>>> Listing files in a directory recursively")
    folder_path = "benchmark-datasets/dev_set"
    files = list_hf_dir_files(root_dir=folder_path)
    print(f"Files in {folder_path}: {files}")
    print(f"Number of files in {folder_path}: {len(files)}")

    # Check if a directory exists
    print("\n>>> Checking if a directory exists")
    subfolder = "benchmark-datasets/dev_set/240130.1_dev/MTG/MTG_30860/mc_plaza_0"
    print(f"Does {subfolder} exist? {is_hf_dir_exists(subfolder)}")

    # Download a directory
    print("\n>>> Downloading a directory")
    download_dir = 'benchmark-datasets/dev_set/240130.1_dev/MTG/MTG_30860/mc_plaza_0'
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading {download_dir} to {temp_dir}")
        downloaded_files_path = download_hf_dir(subfolder=download_dir, local_dir=temp_dir)
        print(f"Downloaded files: {downloaded_files_path}")
        print(f"Number of downloaded files: {len(downloaded_files_path)}")


if __name__ == '__main__':
    main()
