"""
Copyright (c) Microsoft Corporation. All rights reserved.

This module contains functions to download data from Azure blob storage to the local machine.
"""
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Union, Optional, Literal

NOTSOFAR_STORAGE_ACCOUNT_URL = 'https://notsofarsa.blob.core.windows.net'

_LOG = logging.getLogger('azure_storage')


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

    if os.path.exists(local_output_dir) and not overwrite:
        _LOG.info(f'{destination_dir} already exists, skipping download')
        return local_output_dir

    _LOG.info(f'downloading `{azure_source_dir}` from container `{container_name}` to `{local_output_dir}`')
    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'downloading to temp dir first: {temp_dir}')
        os.makedirs(temp_dir, exist_ok=True)
        command = (f'az storage copy --recursive --only-show-errors '
                   f'--destination {temp_dir} '
                   f'--source {NOTSOFAR_STORAGE_ACCOUNT_URL}/{container_name} ')

        if azure_source_dir:
            command += f'--include-path {azure_source_dir.rstrip("/")}'
        try:
            start_time = time.time()
            _LOG.debug(f'command: {command}')
            subprocess.run(command, shell=True, check=True)
            _LOG.info(f'download completed successfully, time: {time.time() - start_time:.0f} seconds')
        except subprocess.CalledProcessError as e:
            _LOG.error(f'failed to download `{azure_source_dir}` '
                       f'from `{container_name}` to `{local_output_dir}`: {e}')
            return None

        if os.path.exists(destination_dir) and overwrite:
            _LOG.debug(f'Deleting existing destination dir: {destination_dir}')
            shutil.rmtree(destination_dir)

        temp_output_dir = (os.path.join(temp_dir, container_name, azure_source_dir)
                           .replace('\\', os.sep)
                           .replace('/', os.sep))
        shutil.move(temp_output_dir, local_output_dir)
    return local_output_dir


def download_meeting_subset(subset_name: Literal['train_set', 'dev_set', 'eval_set'],
                            version: str, destination_dir: Union[str, Path],
                            overwrite: bool = False) -> Optional[str]:
    """
    Download a subset of the meeting dataset to the destination directory.
    The subsets and versions available will be updated in:
        https://www.chimechallenge.org/current/task2/index

    Args:
        subset_name: name of split to download (dev_set / eval_set / train_set)
        version: version to download (240103g / etc.). it's best to use the latest.
        destination_dir: path to the directory where files will be downloaded.
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)


    Latest available datsets:

    # dev_set, no GT available. submit your systems to leaderboard to measure WER.
    res_dir = download_meeting_subset(subset_name='dev_set', version='240130.1_dev', destination_dir=...)

    # train_set, with GT for training models.
    res_dir = download_meeting_subset(subset_name='train_set', version='240130.1_train', destination_dir=...)


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
    azure_dir = f'{f"{pattern}" if pattern is not None else ""}'
    return download_blob_container_dir(azure_source_dir=azure_dir, destination_dir=destination_dir,
                                       container_name=container_name, overwrite=overwrite,
                                       keep_structure=True)


def main():
    """
    Usage example
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] [%(name)s]  %(message)s')

    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'created temp dir: {temp_dir}')

        models_path = download_models(
            destination_dir=os.path.join(temp_dir, 'models'), pattern='notsofar/conformer0.5/mc')
        print(models_path)

        dev_set_dir = download_meeting_subset(
            subset_name='dev_set', version='240130.1_dev',
            destination_dir=os.path.join(temp_dir, 'meeting_data'))
        print(dev_set_dir)

        train_set_path = download_simulated_subset(
            version='v1.4', volume='1000hrs', subset_name='train',
            destination_dir=os.path.join(temp_dir, 'simulated_train'))
        print(train_set_path)


if __name__ == '__main__':
    main()


