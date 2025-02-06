"""
This module contains functions to download the NOTSOFAR dataset and models.
"""
import os
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Union, Optional, Literal

from utils.logging_def import get_logger
from utils.hugging_face_helper import download_hf_dir

_LOG = get_logger('notsofar_dataset')


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

    # dev-set-2, no GT available. Submit your systems to leaderboard to measure WER.
    # dev-set-2 includes mostly new participants compared to the training sets and dev-set-1.
    res_dir = download_meeting_subset(subset_name='dev_set', version='240415.2_dev', destination_dir=...)

    # training set: first and second train-set batches and dev-set-1 (GT unveiled) combined.
    # dev-set-1 and the training sets have significant participant overlap. Use dev-set-2 for development.
    res_dir = download_meeting_subset(subset_name='train_set', version='240501.1_train', destination_dir=...)


    Previous versions:

    # this dataset is identical to the updated "240501.1_train" except it includes some faulty multi-channel
    # devices with replicated channels that have been removed in the newer version.
    res_dir = download_meeting_subset(subset_name='train_set', version='240415.1_train', destination_dir=...)


    # dev-set-1, no GT available. Previous leaderboard was used to measure WER.
    res_dir = download_meeting_subset(subset_name='dev_set', version='240208.2_dev', destination_dir=...)

    # first and second train-set batches combined, with GT for training models.
    res_dir = download_meeting_subset(subset_name='train_set', version='240229.1_train', destination_dir=...)

    # first train-set batch, with GT for training models.
    res_dir = download_meeting_subset(subset_name='train_set', version='240208.2_train', destination_dir=...)


    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    set_type = 'benchmark-datasets'
    _LOG.info(f'Downloading {set_type} subset: {subset_name}, version: {version}')

    destination_dir = Path(destination_dir)
    if overwrite and destination_dir.exists():
        shutil.rmtree(destination_dir)

    hf_subfolder = f'{set_type}/{subset_name}/{version}/MTG'
    download_hf_dir(subfolder=hf_subfolder, local_dir=destination_dir)
    _LOG.info(f'Download completed, download dir: {destination_dir}')

    local_dir = destination_dir / hf_subfolder
    return str(local_dir) if local_dir.exists() else None


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
    _LOG.info(f'Downloading simulated subset: {subset_name}, version: {version}, volume: {volume}')
    set_type = 'css-datasets'
    destination_dir = Path(destination_dir)
    if overwrite and destination_dir.exists():
        shutil.rmtree(destination_dir)

    hf_subfolder = f'{set_type}/{version}/{volume}/{subset_name}'
    download_hf_dir(subfolder=hf_subfolder, local_dir=destination_dir)
    _LOG.info(f'Download completed: {subset_name}, version: {version}, volume: {volume}')
    return str(destination_dir) if destination_dir.exists() else None


def download_models(destination_dir: str, 
                    set_type: str = 'css-models',
                    version: Literal['conformer0.5', 'conformer1.0'] = 'conformer1.0',
                    pattern: Optional[str] = None, overwrite: bool = False) -> Optional[str]:
    """
    Download the models to the destination directory
    Args:
        destination_dir: path to destination directory to download the models to
        version: version of the models to download (conformer0.5 / conformer1.0), default: conformer1.0
        pattern: pattern to match the models to download.
            (e.g. 'mc' will download all notsofar baseline mc models).
        overwrite: whether to override the output file if it already exists
                   (warning!: if true, will delete the entire destination_dir if it exists)
    Returns:
        a string indicates the output directory path, or None if the download failed
    """
    _LOG.info(f'Downloading models: version: {version}, pattern: {pattern}')
    destination_dir = Path(destination_dir)
    models_subdir = f'{set_type}/notsofar/{version}'
    models_local_dir = destination_dir / models_subdir
    if overwrite and models_local_dir.exists():
        shutil.rmtree(models_local_dir)

    download_hf_dir(subfolder=f'{models_subdir}{"/" + pattern if pattern else ""}', local_dir=destination_dir)
    _LOG.info(f'Download completed: models version {version}, pattern: {pattern}')
    return str(models_local_dir) if models_local_dir.exists() else None


def main():
    """
    Usage example for downloading the NOTSOFAR dataset and models.
    """
    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as temp_dir:
        _LOG.info(f'Temp dir: {temp_dir}')
        _LOG.info('Downloading NOTSOFAR dataset and models...')

        _LOG.info('Downloading meeting subset')
        dev_set_dir = download_meeting_subset(
            subset_name='dev_set', version='240208.2_dev', # dev-set is without GT for now
            destination_dir=os.path.join(temp_dir, 'meeting_data'))
        _LOG.info(f'Dev set dir: {dev_set_dir}')

        _LOG.info('Downloading models')
        models_dir = download_models(destination_dir=os.path.join(temp_dir, 'models'), pattern='mc')
        _LOG.info(f'Models dir: {models_dir}')


if __name__ == '__main__':
    main()
