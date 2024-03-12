"""
This module contains functions to download data from Azure blob storage to the local machine.
"""
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Union, Optional, Literal

from utils.logging_def import get_logger

NOTSOFAR_STORAGE_ACCOUNT_URL = 'https://notsofarsa.blob.core.windows.net'

AZCOPY_FILE_NAME_MAP = {'windows': 'azcopy.exe', 'linux': 'azcopy', 'macos': 'azcopy_mos'}
AZCOPY_URL = 'https://notsofarsa.blob.core.windows.net/assets/azcopy/{}'

_LOG = get_logger('azure_storage')


def download_blob_container_dir(azure_source_dir: str, destination_dir: str, container_name: str,
                                keep_structure: bool = False, overwrite: bool = False) -> Optional[str]:
    """
    Download a directory from the container to the given output directory
    Args:
        azure_source_dir: Azure blob directory to download from
        destination_dir: path to destination directory to download to
        container_name: Azure container name
        keep_structure: whether to keep the Azure directory structure in the destination directory
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    azure_dir_path = Path(container_name, azure_source_dir or '')

    local_output_dir = Path(destination_dir)
    if keep_structure:
        local_output_dir = local_output_dir / azure_source_dir

    if local_output_dir.is_dir() and not overwrite:
        _LOG.info(f'{destination_dir} already exists, skipping download')
        return str(local_output_dir)

    _LOG.info(f'Initiating download from `{azure_dir_path}` to `{local_output_dir}`')
    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'Downloading to temp dir first: {temp_dir}')
        azcopy_executable_path = get_azcopy_path()

        azure_dir_url = f'{NOTSOFAR_STORAGE_ACCOUNT_URL}/{azure_dir_path.as_posix()}'
        command = f'{azcopy_executable_path} copy {azure_dir_url} {temp_dir} --recursive'
        _LOG.debug(f'Executing command: {command}')
        try:
            start_time = time.time()
            subprocess.run(command, shell=True, check=True)
            _LOG.info(f'download completed successfully, time: {time.time() - start_time:.0f} seconds')
        except subprocess.CalledProcessError as e:
            _LOG.error(f'Failed to download `{azure_dir_path}` to `{local_output_dir}`: {e}')
            return None

        if local_output_dir.is_dir() and overwrite:
            _LOG.debug(f'Deleting existing destination dir: {destination_dir}')
            shutil.rmtree(str(local_output_dir))

        download_dir_path = Path(temp_dir, azure_dir_path.name)
        _LOG.info(f'moving from temp: {download_dir_path} to local output dir: {local_output_dir}')
        return shutil.move(str(download_dir_path), str(local_output_dir))


def get_azcopy_path() -> str:
    """
    Locates the 'azcopy' executable in the system or deploys it locally if not found.
    Returns the path to the executable.
    Raises FileNotFoundError if the deployment fails.
    Returns:
        path to the azcopy executable
    """
    azcopy_path = shutil.which('azcopy')
    if azcopy_path is None:
        project_root = Path(__file__).resolve().parent.parent
        deployment_dir = project_root / 'artifacts' / 'tools' / 'azcopy'
        deployment_dir.mkdir(parents=True, exist_ok=True)
        os_type = platform.system().replace('Darwin', 'macos').lower()
        azcopy_filename = AZCOPY_FILE_NAME_MAP[os_type]
        azcopy_path = deployment_dir / azcopy_filename
        if os.path.isfile(azcopy_path):
            _LOG.debug(f'AzCopy found at: {azcopy_path}')
            return str(azcopy_path)
        else:
            _LOG.info('AzCopy not found, deploying it to the local machine')
            azcopy_url = AZCOPY_URL.format(azcopy_filename)
            command = f'curl -L {azcopy_url} -o {azcopy_path}'
            subprocess.run(command, shell=True, check=True)
            if not azcopy_path.is_file():
                error_message = f'Failed to deploy azcopy to: {azcopy_path}'
                _LOG.error(error_message)
                raise FileNotFoundError(error_message)
            if os_type in ['linux', 'macos']:
                _LOG.info(f'Gives execution permission to: {azcopy_path}')
                azcopy_path.chmod(0o755)
            _LOG.debug(f'AzCopy deployed to: {azcopy_path}')
    else:
        _LOG.debug(f'AzCopy found at: {azcopy_path}')
    return azcopy_path


def download_meeting_subset(subset_name: Literal['train_set', 'dev_set', 'eval_set'],
                            version: str, destination_dir: Union[str, Path],
                            overwrite: bool = False) -> Optional[str]:
    """
    Downloads a subset of the NOTSOFAR recorded meeting dataset.

    The subsets will be released according to the timeline in:
        https://www.chimechallenge.org/current/task2/index#dates

    Args:
        subset_name: name of split to download (dev_set / eval_set / train_set)
        version: version to download (240103g / etc.). it's best to use the latest.
        destination_dir: path to the directory where files will be downloaded.
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)


    Latest available versions:

    # dev_set, no GT available. submit your systems to leaderboard to measure WER.
    res_dir = download_meeting_subset(subset_name='dev_set', version='240208.2_dev', destination_dir=...)

    # first and second train-set batches combined, with GT for training models.
    res_dir = download_meeting_subset(subset_name='train_set', version='240229.1_train', destination_dir=...)



    Previous versions:

    # first train-set batch, with GT for training models.
    res_dir = download_meeting_subset(subset_name='train_set', version='240208.2_train', destination_dir=...)


    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'benchmark-datasets'
    azure_dir = f'{subset_name}/{version}/MTG'
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite,
                                       keep_structure=True)


def download_simulated_subset(version: str, volume: Literal['200hrs', '1000hrs'],
                              subset_name: Literal['train', 'val'], destination_dir: str,
                              overwrite: bool = False) -> Optional[str]:
    """
    Download the simulated dataset to the destination directory
    Args:
        version: version of the train data to download (v1 / v1.1 / v1.2 / v1.3 / etc.)
        volume: volume of the train data to download (200hrs / 1000hrs)
        subset_name: train data type to download (train / val)
        destination_dir: path to the directory where files will be downloaded.
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)


    Latest available datasets:

    # 1000 hours
    train_set_path = download_simulated_subset(version='v1.5', volume='1000hrs', subset_name='train',
            destination_dir=...)
    val_set_path = download_simulated_subset(version='v1.5', volume='1000hrs', subset_name='val',
            destination_dir=...)

    # 200 hours subset
    train_set_path = download_simulated_subset(version='v1.5', volume='200hrs', subset_name='train',
            destination_dir=...)
    val_set_path = download_simulated_subset(version='v1.5', volume='200hrs', subset_name='val',
            destination_dir=...)


    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'css-datasets'
    azure_dir = '/'.join([version, volume, subset_name])
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite,
                                       keep_structure=True)


def download_models(destination_dir: str, pattern: Optional[str] = None,
                    overwrite: bool = False) -> Optional[str]:
    """
    Download the models to the destination directory
    Args:
        destination_dir: path to destination directory to download the models to
        pattern: pattern to match the models to download.
            (e.g. 'notsofar/mc' will download all notsofar baseline mc models).
            To review all available models, view the container using Azure CLI or Azure Storage Explorer.
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'css-models'
    azure_dir = pattern or ''
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite,
                                       keep_structure=True)


def main():
    """
    Usage example
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'created temp dir: {temp_dir}')

        dev_set_dir = download_meeting_subset(
            subset_name='dev_set', version='240208.2_dev', # dev-set is without GT for now
            destination_dir=os.path.join(temp_dir, 'meeting_data'))
        print(dev_set_dir)

        train_set_path = download_simulated_subset(
            version='v1.5', volume='1000hrs', subset_name='train',
            destination_dir=os.path.join(temp_dir, 'simulated_train'))
        print(train_set_path)


if __name__ == '__main__':
    main()


