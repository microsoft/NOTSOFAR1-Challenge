import torch
from typing import Dict


class MicShiftAugmentation:
    """
    Augments data by randomly shifting circular microphones 1-6 cyclically while preserving mic 0.
    Assumption: mics are ordered 0..7 in features.
    """

    def __init__(self, seed: int, device: torch.device = torch.device('cpu')):
        self.rgen = torch.Generator(device=device)
        self.rgen.manual_seed(seed)

    def __call__(self, segment_batch: Dict) -> Dict:
        """Performs augmentation on a batch of segments.

        Args:
            segment_batch: a batch of segments, each segment is a dict of tensors. See the SimulatedDataset class for
                the expected keys. Note that the expected tensors are of shape [Batch, T, Mics] or
                [Batch, T, Mics, Spks], depending on the field.

        Returns:
            The batch with the same keys, but with the microphone arrays shifted.
        """

        ignore_keys = ['utterance_id', 't0', 'seg_len', 'gt_spk_activity_scores',]
        # keys that require permutation
        mic_array_keys = ['mixture', 'gt_spk_direct_early_echoes', 'gt_spk_reverb', 'gt_noise']

        not_covered = set(segment_batch) - set(ignore_keys + mic_array_keys)
        assert not not_covered, f'Unexpected keys! add them to ignore_keys, ' \
                                f'or to mic_array_keys and process them: {not_covered}'

        batch_size = segment_batch['mixture'].shape[0]
        shifts = torch.randint(0, 6, (batch_size,), generator=self.rgen, device=self.rgen.device)

        # shift all values by the same offset
        for key in mic_array_keys:
            if key in segment_batch:
                arr = segment_batch[key]
                assert arr.shape[2] == 7, 'expecting 7 microphones at dim 2'

                # Shift all mics except 0
                arr[:, :, 1:] = _batch_roll_dim2(arr[:, :, 1:], shifts)

        return segment_batch


def _batch_roll_dim2(arr, shifts):
    """Rolls the values of the third dimension of a batch of tensors.

    Args:
        arr: The array of shape [Batch, T, Mics] or [Batch, T, Mics, Spks] to roll.
        shifts: The number of shifts to perform for each batch element.

    Returns:
        The rolled array.
    """

    # Add a singleton dimension if needed
    orig_ndim = arr.ndim
    if orig_ndim == 3:
        arr = arr.unsqueeze(-1)

    # Assuming arr of shape [batch_size, mics, T, spks]
    batch_size, t, mics, spks = arr.shape

    # Create a grid of mic indices of the same shape as the input tensor
    indices = torch.arange(mics, device=arr.device)[None, None, :, None].repeat(batch_size, t, 1, spks)

    # Verify that shifts is a vector of the same size as the batch size
    assert shifts.shape == (batch_size,), f'Expecting shifts to be a vector of the same size as the batch size!'

    # Adjust indices for the shifts, ensuring wrapping around
    indices = (indices - shifts[:, None, None, None]) % mics

    # Gather the values from the input tensor according to the shifted indices.
    # The following will result in:
    #   rolled[batch][t][mic][spk] = arr[batch][t][ indices[batch][t][mic] ][spk].
    rolled = torch.gather(arr, 2, indices)

    # Remove the singleton dimension if needed
    if orig_ndim == 3:
        rolled = rolled.squeeze(-1)

    return rolled


def test_batch_roll_dim2():
    batch_size = 32
    t = 48000
    mics = 7
    spks = 3

    for with_spks in [True, False]:
        for i in range(100):
            m = torch.rand(batch_size, t, mics, spks) if with_spks else torch.rand(batch_size, t, mics)

            shifts = torch.randint(0, 6, (batch_size,))

            # Fast version
            r1 = m.clone()
            r1[:, :, 1:] = _batch_roll_dim2(m[:, :, 1:], shifts)

            # Slow version
            r2 = m.clone()
            for b in range(batch_size):
                r2[b, :, 1:] = torch.roll(m[b, :, 1:], shifts=shifts[b].item(), dims=1)

            # Check that the results are the same
            assert (r1 == r2).all(), 'Failed!'

    print('batch_roll_dim2 test passed!')


if __name__ == '__main__':
    test_batch_roll_dim2()
