"""
This module contains functions to deploy data from Azure blob storage to the local machine.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Union, Optional

NOTSOFAR_STORAGE_ACCOUNT_URL = 'https://notsofarsa.blob.core.windows.net'

_LOG = logging.getLogger('data_deployment')


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
    local_output_dir = destination_dir
    if keep_structure:
        local_output_dir = os.path.join(destination_dir, azure_source_dir).replace(
            '\\', os.sep).replace('/', os.sep)

    if os.path.exists(destination_dir) and not overwrite:
        _LOG.debug(f'{destination_dir} already exists, skipping download')
        return local_output_dir

    _LOG.info(f'downloading `{azure_source_dir}` from container `{container_name}` to `{local_output_dir}`')
    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.debug(f'created temp dir: {temp_dir}')
        os.makedirs(temp_dir, exist_ok=True)
        command = (f'az storage copy --recursive --only-show-errors '
                   f'--destination {temp_dir} '
                   f'--source {NOTSOFAR_STORAGE_ACCOUNT_URL}/{container_name} '
                   f'--include-path {azure_source_dir.rstrip("/")}')
        try:
            start_time = time.time()
            _LOG.debug(f'command: {command}')
            subprocess.run(command, shell=True, check=True)
            _LOG.info(f'download completed successfully, time: {time.time() - start_time:.0f} seconds')
        except subprocess.CalledProcessError as e:
            _LOG.error(f'failed to download `{azure_source_dir}` from `{container_name}` to `{local_output_dir}`: {e}')
            return None

        if os.path.exists(destination_dir) and overwrite:
            _LOG.debug(f'Deleting existing destination dir: {destination_dir}')
            shutil.rmtree(destination_dir)

        temp_output_dir = os.path.join(temp_dir, container_name, azure_source_dir).replace('\\', os.sep).replace('/', os.sep)
        shutil.move(temp_output_dir, local_output_dir)
    return local_output_dir


def deploy_benchmark_dataset(set_type: str, version: str, destination_dir: Union[str, Path],
                             overwrite: bool = False) -> Optional[str]:
    """
    Deploy the benchmark dataset to the destination directory
    Args:
        set_type: benchmark dataset type to deploy (dev_set / eval_set / train_set)
        version: version of the benchmark dataset to deploy (240103g / etc.)
        destination_dir: path to destination directory to download to
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'benchmark-datasets'
    azure_dir = f'{set_type}/{version}/MTG'
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite, keep_structure=True)


def deploy_simulated_dataset(version: str, volume: str, set_type: str, destination_dir: str,
                             overwrite: bool = False) -> Optional[str]:
    """
    Deploy the simulated dataset to the destination directory
    Args:
        version: version of the train data to deploy (v1 / v1.1 / v1.2 / v1.3 / etc.)
        volume: volume of the train data to deploy (200hrs / 1000hrs)
        set_type: train data type to deploy (train / val)
        destination_dir: path to destination directory to download to
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'css-datasets'
    azure_dir = '/'.join([version, volume, set_type])
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite, keep_structure=True)


def deploy_models(destination_dir: str, pattern: Optional[str] = None, overwrite: bool = False) -> Optional[str]:
    """
    Deploy the models to the destination directory
    Args:
        destination_dir: path to destination directory to download the models to
        pattern: pattern to match the models to download (e.g. 'espnet/mc' will download all espnet mc models).
                 to review all available models, view the container using Azure CLI or Azure Storage Explorer.
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    container_name = 'css-models'
    azure_dir = f'css_model{f"/{pattern}" if pattern is not None else ""}'
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite, keep_structure=True)


def main():
    """
    Usage example
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] [%(name)s]  %(message)s')

    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'created temp dir: {temp_dir}')
        dev_set_dir = deploy_benchmark_dataset(
            set_type='dev_set', version='240103g', destination_dir=os.path.join(temp_dir, 'benchmark'))
        print(dev_set_dir)

        train_set_path = deploy_simulated_dataset(
            version='v1', volume='200hrs', set_type='train', destination_dir=os.path.join(temp_dir, 'train'))
        print(train_set_path)

        models_path = deploy_models(
            destination_dir=os.path.join(temp_dir, 'models'), pattern='espnet/mc')
        print(models_path)


if __name__ == '__main__':
    main()
