import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Union, Optional, List

from tqdm import tqdm
from huggingface_hub import HfApi


api = HfApi()
NOTSOFAR_HF_REPO_ID = "microsoft/NOTSOFAR"


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
        files = api.list_repo_tree(repo_id=NOTSOFAR_HF_REPO_ID, repo_type="dataset", path_in_repo=subfolder)
        return True if files else False
    except Exception as e:
        return False


def list_hf_dir_files(root_dir: Union[str, Path]) -> List[str]:
    """
    Iterate over all files in the Hugging Face repository using hf_hub_list
    Args:
        root_dir: root directory of the repository
        repo_id: Hugging Face repository ID
    """
    root_dir = str(root_dir)
    assert is_hf_dir_exists(root_dir), f"Cannot find {root_dir} in the Hugging Face repository"
    return [val.path for val in api.list_repo_tree(repo_id=NOTSOFAR_HF_REPO_ID, repo_type="dataset",
                                                   path_in_repo=root_dir)]


def download_hf_file(file_path: str, local_dir: Path, pbar: Optional[tqdm]):
    """
    Download a file from the Hugging Face repository.

    Args:
        file_path: path of the file in the repository
        local_dir: local directory to download the file to
        pbar: tqdm progress bar object (optional)
    """
    local_file_path = local_dir / file_path
    os.makedirs(local_file_path.parent, exist_ok=True)
    api.hf_hub_download(repo_id=NOTSOFAR_HF_REPO_ID, filename=file_path,
                        repo_type="dataset", local_dir=local_file_path.parent)
    pbar.update(1)


def download_hf_dir(subfolder: str, local_dir: Union[str, Path], max_workers: int = os.cpu_count()):
    """
    Download all files in a subfolder of the Hugging Face repository

    Args:
        subfolder: path in the repository to download
        local_dir: local directory to download the files to
        max_workers: number of workers to use for downloading, defaults to number of CPUs

    Returns:

    """
    assert is_hf_dir_exists(subfolder), f"Subfolder {subfolder} does not exist in the Hugging Face repository"
    assert isinstance(local_dir, (str, Path)), "local_dir should be a string or Path object"
    assert isinstance(max_workers, int), "max_workers should be an integer"

    local_dir = Path(local_dir)
    files = list_hf_dir_files(root_dir=subfolder)

    with tqdm(total=len(files), desc="Downloading", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_hf_file, file, local_dir, pbar) for file in files]
            wait(futures)


def main():
    """
    Usage example for the Hugging Face helper functions
    """
    folder_path = "benchmark-datasets/dev_set"
    files = list_hf_dir_files(root_dir=folder_path)
    print(files)

    download_dir = 'benchmark-datasets/dev_set/240130.1_dev/MTG/MTG_30860/mc_plaza_0'
    dst_dir = 'C:\dev\Temp\HF DOWNLOAD_TEST'
    download_hf_dir(subfolder=download_dir, local_dir=dst_dir)


if __name__ == '__main__':
    main()
