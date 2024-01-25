""" Simulated dataset loader for NOTSOFAR and related pieces """
from typing import Sequence, Callable, Optional, Tuple, Union, List, Dict
from pathlib import Path
import glob
import math
import json
import tarfile

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate # type: ignore
import torch.distributed as dist


class SimulatedDataset(Dataset):
    """
    A PyTorch Dataset for loading simulated data efficiently,
    compatible with Distributed Data Parallel (DDP) training.

    During loading, segments are read in random access with a seek operation.

    Data is packaged in tar files. Each tar file contains several utterances,
    and the attributes of these utterances are saved in separate files within the tar file.

    Assumptions about stored data:
        - Attributes of the utterances can be either arrays or scalars.
        - Array attributes must have equal length (the size of the first dimension i.e., len(array)),
            within a single utterance.
        - Arrays are saved as int16 with a corresponding *_scale attribute for rescaling to float.
        - Audio data is stored at a sample rate of 16,000 Hz.
    """

    def __init__(self, dataset_path: str, segment_split_func, transform_fns: Sequence[Callable] = (),
                 seed: int = 25486541, sample_frac: float = 1.0, max_urls: Optional[int] = None,
                 fs: int = 16000, single_channel=False, max_spks=3, needed_columns: Optional[List[str]] = None):
        super().__init__()
        self.split_func = segment_split_func
        self.transform_fns = list(transform_fns)
        # This is used to randomize various stages:
        #   splitting to segments, transforms, and sample_frac (when sample_frac < 1).
        self.rstate = np.random.RandomState(seed)
        self.fs = fs
        self.total_len_sec = None
        self.dataset = self._create_dataset(dataset_path, sample_frac, max_urls)
        self.single_channel = single_channel
        self.max_spks = max_spks

        # If needed_columns was not specified, use all columns
        all_columns = ['mixture', 'gt_spk_activity_scores', 'gt_spk_direct_early_echoes', 'gt_spk_reverb', 'gt_noise']
        if needed_columns is None:
            needed_columns = all_columns
        else:
            needed_columns = list(needed_columns)  # make a copy
            assert set(needed_columns).issubset(all_columns), \
                f'Invalid column names in needed_columns: {set(needed_columns) - set(all_columns)}'

        # Add to needed_columns the *_scale columns for the columns that require scaling
        require_scale = ['mixture', 'gt_spk_direct_early_echoes', 'gt_spk_reverb', 'gt_noise']
        needed_columns += [f'{col}_scale' for col in needed_columns if col in require_scale]
        self.needed_columns = needed_columns

    def _create_dataset(self, save_dir, sample_frac, max_urls) -> List[Dict]:
        """ Parse data folder and create a global meta-dataset """

        tar_files = glob.glob(f"{save_dir}/*.tar")
        map_files = glob.glob(f"{save_dir}/*.map")
        assert (bool(tar_files) + bool(map_files)) == 1, 'expecting either tar files or individual utterances'
        is_tar = len(tar_files) > 0
        if is_tar:
            map_files = tar_files

        def read_utterance_map(path: str):
            if is_tar:
                with tarfile.open(url, "r") as tar_file:
                    utt_map = tar_file.extractfile("utterances.map").read()
            else:
                with open(path, "rb") as file:
                    utt_map = file.read()
            return json.loads(utt_map)

        # Sort the map files to ensure deterministic behavior
        map_files.sort()

        # The dataset is split into partitions. Each partition's utterance metadata is stored in *.map files.
        urls = [Path(f).absolute().as_posix() for f in map_files]
        urls = self.rstate.choice(urls, math.ceil(len(urls) * sample_frac), replace=False)
        urls = urls if max_urls is None else urls[:min(max_urls, len(urls))]
        dataset = []
        self.total_len_sec = 0
        for url in urls:
            utterances_map = read_utterance_map(url)
            for utterance, utt_length in utterances_map.items():
                self.total_len_sec += utt_length / self.fs
                # Split utterance into segments to determine the total number of segments it contributes.
                segments, _ = self.split_func(utt_length)
                for index in range(len(segments)):
                    dataset.append({"id": utterance, "index": index, "url": url, "size": utt_length})
        return dataset

    def _extract_segment(self, utterance_uid, dir_path: Union[tarfile.TarFile, Path], offset = 0,
                         seg_len = None) -> Dict:
        """ Extracts a segment from the tar file """

        def seek_segment(filename: str, offset: Optional[int] = None, row_size: Optional[int] = None,
                         seg_len: Optional[int] = None):
            if isinstance(dir_path, tarfile.TarFile):
                extracted = dir_path.extractfile(filename)
                if offset is None:
                    return extracted.read()
                extracted.seek(offset * row_size)
                data = extracted.read() if seg_len is None else extracted.read(seg_len * row_size)
                return data
            else:
                file_path = dir_path / filename
                with file_path.open('rb') as file:
                    if offset is None:
                        return file.read()
                    file.seek(offset * row_size)
                    data = file.read() if seg_len is None else file.read(seg_len * row_size)
                    return data

        metadata = seek_segment(f"{utterance_uid}.json")
        metadata = json.loads(metadata)
        columns = metadata['columns']
        assert metadata['index_value'] == utterance_uid
        d = {'utterance_id': utterance_uid, 't0': offset, 'seg_len': seg_len}
        for column in self.needed_columns:
            if 'values' in columns[column]:
                d[column] = eval(columns[column]['values'])
            else:
                data = seek_segment(f"{metadata['index_value']}.{column}",
                                       offset=offset, row_size=columns[column]['row_size'], seg_len=seg_len)
                array = np.frombuffer(data, dtype=columns[column]['dtype'])

                # Make a copy of the array to allow its conversion to a torch tensor without getting the following
                # warning repeatedly:
                # venv\lib\site-packages\torch\utils\data\_utils\collate.py:172: UserWarning: The given NumPy array is
                # not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor
                # will result in undefined behavior. You may want to copy the array to protect its data or make it
                # writable before converting it to a tensor. This type of warning will be suppressed for the rest of
                # this program. (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:205.)
                #   return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)
                # TODO: Consider alternative solutions to avoid the copy.
                array = array.copy()

                shape = columns[column]['shape']
                if seg_len is not None:
                     shape[0] = seg_len
                array = array.reshape(shape)
                d[column] = array
        return d

    def _pad_to_max_spks(self, seg: Dict):
        """ Pads the arrays to max_spks """

        # Only the following columns require padding
        for k in ['gt_spk_activity_scores', 'gt_spk_direct_early_echoes', 'gt_spk_reverb']:
            if k in seg:
                assert seg[k].shape[-1] <= self.max_spks, \
                    f'Expected {k} to have at most {self.max_spks} speakers, got {seg[k].shape[-1]}!'

                # Skip if padding is not needed
                if seg[k].shape[-1] == self.max_spks:
                    continue

                # Pad with -1 for gt_spk_activity_scores, and 0.0 for the other columns
                pad_value = -1 if k == 'gt_spk_activity_scores' else 0.0

                # Pad the last dimension to max_spks with zeros
                pad_per_dim = [(0, 0)] * (seg[k].ndim-1) + [(0, self.max_spks - seg[k].shape[-1])]
                seg[k] = np.pad(seg[k], pad_per_dim, mode='constant', constant_values=pad_value)

        # Verify that the type of gt_spk_activity_scores remained int8
        assert 'gt_spk_activity_scores' not in seg or seg['gt_spk_activity_scores'].dtype == np.int8, \
            f'Expected gt_spk_activity_scores to be int8, got {seg["gt_spk_activity_scores"].dtype}!'

    def get_length_seconds(self):
        """
        Returns the total length of the dataset in seconds.

        Note that the actual combined length of all segments is greater than the total duration
        because the segments are overlapping.
        """
        return self.total_len_sec

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Union[ndarray, float, str]]:
        """
        Returns a single segment from the dataset.

        Args:
            idx (int): Index of the segment to retrieve. It refers to one segment out of the entire dataset
                of segments coming from all utterances.

        Returns: A dictionary containing various components of the audio segment, with the following keys:
            't0' (int): The start time of the segment in samples.
            'seg_len' (int): The length of the segment in samples.

            'mixture' (ndarray): A 2D array with dimensions [T, Mics], representing the input mixture signal.
                'T' is the number of samples, and 'Mics' is the number of microphones.

            'gt_spk_activity_scores' (ndarray): A 2D array [T, Max_spks] containing the ground truth (GT)
                speech activity scores. Values:
                -1: Not speaking.
                 0: Borderline energy. Represents a transition region between speech and silence.
                 1: Speaking.
                'Max_spks' represents the maximum number of speakers.

            'gt_spk_direct_early_echoes' (ndarray): A 3D array [T, Mics, Max_spks] representing the
                direct path and early echoes component for each speaker.

            'gt_spk_reverb' (ndarray): A 3D array [T, Mics, Max_spks] containing the reverberation component
                for each speaker.

            'gt_noise' (ndarray): A 2D array [T, Mics] representing the noise component of the segment.
                This includes both stationary and non-stationary noise.

            'utterance_id' (str): A unique identifier for the source utterance of the segment.
                TODO: Consider adding speaker ID labels to further enhance the dataset's utility.

            Note: The GT (ground truth) components sum up to the mixture (up to numerical error):
                  mixture ≈ gt_spk_direct_early_echoes.sum(-1) + gt_spk_reverb.sum(-1) + gt_noise
        """
        utterance = self.dataset[idx]
        rand_seed = self.rstate.randint(int(1e9))
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = (rand_seed, rank)

        # Split utterance into segments and return segment at index - utterance['index'].
        # The seed is randomized so the resulting segments will be somewhat shifted each time,
        # providing data augmentation. Note that the total number of segments per utterance remains constant.
        segments, seg_len = self.split_func(utterance['size'], seed=seed)
        data_part = Path(utterance['url'])
        if data_part.suffix == '.tar':
            with tarfile.open(utterance['url'], 'r') as tar_file:
                seg = self._extract_segment(utterance['id'], tar_file, segments[utterance['index']], seg_len)
        else:
            # '.map' suffix
            seg = self._extract_segment(utterance['id'], data_part.parent, segments[utterance['index']],
                                        seg_len)

        # seg = pd.Series(seg)
        # apply scaling and convert to float32
        scale_columns = [col for col in seg.keys() if col.endswith('_scale')]
        for scale_col in scale_columns:
            orig_col = scale_col[:-len('_scale')]
            assert seg[orig_col].dtype == np.int16
            seg[orig_col] = seg[orig_col].astype(np.float32) / seg[scale_col]

        # drop scale factor columns
        for c in scale_columns:
            del seg[c]

        # If only a single channel is needed, drop the other channels
        if self.single_channel:
            for k in ['mixture', 'gt_spk_direct_early_echoes', 'gt_spk_reverb', 'gt_noise']:
                if k in seg:
                    # TODO: The reference channel number below (==0) needs to be configurable.
                    seg[k] = seg[k][:, 0:1]  # note that the mics dimension will be kept as a singleton

        # apply transforms
        for transform_fn in self.transform_fns:
            seed = (self.rstate.randint(int(1e9)), rank)
            seg = transform_fn(seg, seed)

        # pad some columns to max_spks
        self._pad_to_max_spks(seg)

        return seg


class SegmentSplitter:
    """ Selects random segments from an utterance """

    def __init__(self, min_overlap: int = 50, max_overlap: int = 150, pr_force_align: float = 0.5,
                 desired_segm_len: Union[int, Tuple[int, int]] = 300):
        """
        Args:
            min_overlap: Minimum allowed overlap
            max_overlap: Maximum allowed overlap
            pr_force_align: If after setting the overlap, we are not covering the entire interval,
                this is the probability that we force to align the subset of interval that we cover
                either to right or left. Otherwise, (pr=1-pr_force_align), we just randomize position
                in the valid range.
            desired_segm_len: either the exact length required or a range (tuple) to randomize from.
        """
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.pr_force_align = pr_force_align
        self.desired_segm_len = desired_segm_len

        assert 0. <= self.pr_force_align <= 1.
        assert 0 <= self.min_overlap <= self.max_overlap

    def interval_cover(self, utt_lengths: ndarray, segm_len: int) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Args:
            utt_lengths: For each utterance
            segm_len:

        Returns: steps_num, overlap_samples, uncovered_samples for each interval_len
            If utt_lengths < segm_len return 0, 0, interval_len  (ie, utterance too short and is not covered)
        """
        assert segm_len > self.max_overlap
        steps_num = np.maximum(
            0, np.floor((utt_lengths - self.max_overlap) / (segm_len - self.max_overlap)).astype(int))
        overlap = np.zeros_like(steps_num)
        uncovered_samples = utt_lengths.copy()
        eq1 = (steps_num == 1)
        gt1 = (steps_num > 1)  # Greater than 1
        # All the following calculations are done only for cases where steps_num > 0:
        overlap_for_full_cover_gt1 = np.ceil((steps_num[gt1] * segm_len - utt_lengths[gt1]
                                              ) / (steps_num[gt1] - 1)).astype(int)
        overlap_at_gt1 = np.maximum(overlap_for_full_cover_gt1, self.min_overlap)
        assert np.all(overlap_at_gt1 <= self.max_overlap)
        cover_samples_at_gt1 = segm_len + (segm_len - overlap_at_gt1) * (steps_num[gt1] - 1)

        overlap[gt1] = overlap_at_gt1
        uncovered_samples[gt1] = utt_lengths[gt1] - cover_samples_at_gt1
        uncovered_samples[eq1] = utt_lengths[eq1] - segm_len
        assert np.all(uncovered_samples >= 0)

        return steps_num, overlap, uncovered_samples

    def shuffled_segments(self, utt_lengths: ndarray, shuffle: bool,
                          epoch_ind: int, rand_seed: int) -> Tuple[List[Tuple[int, int]], int]:
        """ Covers each utterance by one or more segments.
        Parameters define how to cover the entire utterances (e.g., target overlaps)
        and what to do if utterance cannot be covered (how to randomly select subset of the utterance).
        Optional final step: shuffle all resulting segments.

        Args:
            utt_lengths: Length for each source utterance
            shuffle:
            epoch_ind:
            rand_seed:

        Returns:
            tuple:
                - List of (utt_index, first_sample) tuples, possibly shuffled,
                - the selected segment length
        """
        rstream = np.random.RandomState((rand_seed, epoch_ind, 0))

        segm_len = (rstream.randint(*self.desired_segm_len) if isinstance(self.desired_segm_len, tuple)
                    else self.desired_segm_len)

        n = utt_lengths.size
        steps_num, overlap, uncovered_samples = self.interval_cover(utt_lengths, segm_len)
        force_alignment = (rstream.uniform(0., 1., n) < self.pr_force_align)
        delay_first_rel_uncovered = (force_alignment * (rstream.uniform(0., 1., n) < 0.5)
                                     + (1 - force_alignment) * rstream.uniform(0., 1., n))
        delay_first_sample = np.floor(uncovered_samples * delay_first_rel_uncovered).astype(int)

        segments = []
        # pylint: disable=invalid-name
        for utt_ind, (steps, delay, ov) in enumerate(zip(steps_num, delay_first_sample, overlap)):
            segments.extend([(utt_ind, t0) for t0 in delay + np.arange(steps) * (segm_len - ov)])

        for utt_ind, t0 in segments:
            assert t0 + segm_len <= utt_lengths[utt_ind]

        # Random shuffle:
        if shuffle:
            rstream_shuffle = np.random.RandomState((rand_seed, epoch_ind, 1))
            rstream_shuffle.shuffle(segments)

        return segments, segm_len

    def __call__(self, utt_length: int,
                 seed: Optional[Tuple[int, int]] = (39565, 0)) -> Tuple[List[int], int]:
        """
        Splits a single utterance into segments.

        Args:
            utt_length: legth of utterance to split
            seed: seeds random segment selection

        Returns:
            tuple:
                - List[int]: A list of segment starting samples (t0)
                - int: segment length (all the utterance's segments have equal length)
        """

        if seed is None:
            seed = (39565, 0)

        # shuffled_segments support batch processing of multiple utterances, but we're handling a single
        # utterance in this case.
        utt_lengths = np.array([utt_length])

        segments, seg_len = self.shuffled_segments(  # do not shuffle segments (happens at DataLoader level)
            utt_lengths, shuffle=False, rand_seed=seed[0], epoch_ind=seed[1])

        return [s[1] for s in segments], seg_len


if __name__ == "__main__":
    project_root = Path(__file__).parents[2]
    data_path = str(project_root / 'sample_data' / 'css_train_set')

    seg_len_samps = 3 * 16000  # 3 seconds in samples
    seg_split = SegmentSplitter(seg_len_samps // 6, seg_len_samps // 2, 0.5, seg_len_samps)

    dataset = SimulatedDataset(data_path, seg_split, transform_fns=[])

    [print(f'{k}: {v if not isinstance(v, np.ndarray) else v.shape}') for k, v in dataset[0].items()]

    print('\nStats:\n')
    print(f'len(dataset) = {len(dataset)}')
    print(f'length in seconds = {dataset.get_length_seconds()}')


# TODO: Remove the following before release.
class DummySimulatedDataset(Dataset):
    """ A dummy dataset for testing purposes """

    def __init__(self, num_samples=100000, desired_segm_len=48000, max_spks=4):
        super().__init__()
        self.num_samples = num_samples
        self.desired_segm_len = desired_segm_len
        self.max_spks = max_spks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'mixture': np.zeros((self.desired_segm_len, 7), dtype=np.float32),
            'gt_spk_direct_early_echoes': np.zeros((self.desired_segm_len, 7, self.max_spks), dtype=np.float32),
            'gt_noise': np.zeros((self.desired_segm_len, 7), dtype=np.float32),
        }
