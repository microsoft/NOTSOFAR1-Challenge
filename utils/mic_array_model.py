import numpy as np


def multichannel_mic_pos_xyz_cm() -> np.ndarray:
    """
    Returns the mic positions in cm of multichannel devices used in NOTSOFAR.
    The order matches the wav files of multichannel sessions.

    Returns:
        mic_pos_xyz_cm: (7, 3) array of mic positions in cm.
            mic_pos_xyz_cm[0, :] is the center microphone's x,y,z.

    """
    # TODO: finalize these numbers
    mic1_az = 0.
    az_dir_sign = 1.

    mic_pos_xyz_cm = np.empty((7, 3))
    mic_pos_xyz_cm[:] = np.nan
    mic_pos_xyz_cm[0, :] = 0.
    r = 4.25
    for i in range(1, 7):
        mic_pos_xyz_cm[i, 0] = r * np.cos(np.deg2rad(az_dir_sign * 60. * (i - 1) + mic1_az))
        mic_pos_xyz_cm[i, 1] = r * np.sin(np.deg2rad(az_dir_sign * 60. * (i - 1) + mic1_az))
        mic_pos_xyz_cm[i, 2] = 0.
    assert not np.any(np.isnan(mic_pos_xyz_cm))
    return mic_pos_xyz_cm

