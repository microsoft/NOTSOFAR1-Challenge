import numpy as np


def erode(arr: np.ndarray, iters: int):
    assert arr.ndim == 1
    arr_padded = np.pad(arr, iters, mode='constant', constant_values=1)
    return np.lib.stride_tricks.sliding_window_view(arr_padded, 2 * iters + 1).min(1)


def dilate(arr: np.ndarray, iters: int):
    assert arr.ndim == 1
    arr_padded = np.pad(arr, iters, mode='constant', constant_values=0)
    return np.lib.stride_tricks.sliding_window_view(arr_padded, 2 * iters + 1).max(1)


def test_morphology():
    arr = np.array(  [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    eroded = erode(arr, 1)
    assert np.all(eroded == [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    dilated = dilate(arr, 1)
    assert np.all(dilated == [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0])


if __name__ == "__main__":
    test_morphology()