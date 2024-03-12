import librosa
import numpy as np


def make_mvdr(spk_masks, noise_masks, mix_wav = None, mix_stft = None, return_stft=False):
    """

    Args:
        mix_wav: mixture waveform, [Nsamples, Mics] tensor
        spk_masks: [num_spks, F, T] tensor
        noise_masks: [num_noise, F, T] tensor
        mix_stft: mixture STFT, [Mics, F, T] complex tensor
        return_stft: if True, return the STFT of the separated signals.
            Otherwise, return the separated signals in the time domain.

    Returns:

    """
    all_masks = make_wta(spk_masks, noise_masks)  # [num_spks + 1_noise, F, T]
    if mix_stft is None:
        mix_stft=[]
        for i in range(7):
            st=librosa.core.stft(mix_wav[:, i], n_fft=512, hop_length=256)
            mix_stft.append(st)
        mix_stft=np.asarray(mix_stft)  # [Mics, F, T]

    L = np.min([all_masks.shape[-1],mix_stft.shape[-1]])
    mix_stft = mix_stft[:,:,:L]
    all_masks = all_masks[:,:,:L]

    scms = [get_mask_scm(mix_stft, mask) for mask in all_masks]
    spk_scms = np.stack(scms[:-1])  # [num_spks, F, 7, 7]
    noise_scm = scms[-1]  # [F, 7, 7]

    res_per_spk = []
    for i in range(spk_scms.shape[0]):
        # sum SCMs of all other speakers
        other_spks_scm = spk_scms[np.arange(spk_scms.shape[0]) != i].sum(axis=0)
        # add noise and compute beamforming coefficients for the current speaker
        coef = calc_bfcoeffs(noise_scm + other_spks_scm, spk_scms[i])
        res = get_bf(mix_stft, coef)
        res_per_spk.append(res)

    if not return_stft:
        res_per_spk = [librosa.istft(res, hop_length=256) for res in res_per_spk]

    return res_per_spk


def make_wta(spk_masks, noise_masks):
    noise_mask = noise_masks.sum(axis=0, keepdims=True)
    mask = np.vstack([spk_masks, noise_mask])
    mask_max = np.amax(mask, axis=0, keepdims=True)
    mask = np.where(mask==mask_max, mask, 1e-10)
    return mask


def get_mask_scm(mix,mask):
    """Return spatial covariance matrix of the masked signal."""

    Ri = np.einsum('FT,FTM,FTm->FMm',
                   mask, mix.transpose(1,2,0), mix.transpose(1,2,0).conj())
    t1=np.eye(7)
    t2=t1[np.newaxis,:,:]
    Ri+=1e-15*t2
    return Ri  # ,np.sum(mask)


def calc_bfcoeffs(noi_scm,tgt_scm):
    # Calculate BF coeffs.
    num = np.linalg.solve(noi_scm, tgt_scm)
    den = np.trace(num, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]
    den[0]+=1e-15
    W = (num / den)[..., 0]
    return W


def get_bf(mix,W):
    c,f,t=mix.shape
    return np.sum(W.reshape(f,c,1).conj()*mix.transpose(1,0,2),axis=1)
