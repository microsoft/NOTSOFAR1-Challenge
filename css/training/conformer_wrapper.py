import warnings
from dataclasses import dataclass, field, asdict
from css.css_with_conformer.executor.executor import Executor
from css.css_with_conformer.nnet.conformer import ConformerCSS
import torch as th
import torch.nn as nn


# The default values for the ExtractorCfg, ConformerCfg, and CssWithConformerCfg dataclasses were taken from the
# conformer_base (MC) model.
@dataclass
class ExtractorCfg:
    ang_index: str = ''
    frame_hop: int = 256
    frame_len: int = 512
    ipd_cos: bool = False
    ipd_index: str = '1,0;2,0;3,0;4,0;5,0;6,0'
    ipd_mean_normalize: bool = True
    ipd_mean_normalize_version: int = 1
    log_spectrogram: bool = False
    mvn_spectrogram: bool = True
    num_spks: int = 2
    round_pow_of_two: bool = True
    window: str = 'hann'


@dataclass
class ConformerCfg:
    attention_dim: int = 256
    attention_heads: int = 4
    dropout_rate: float = 0.1
    kernel_size: int = 33
    linear_units: int = 1024
    num_blocks: int = 16


@dataclass
class NnetCfg:
    conformer_conf: ConformerCfg = field(default_factory=ConformerCfg)
    in_features: int = 1799
    num_nois: int = 1  # CSS_with_Conformer had 2 noise masks. NOTSOFAR has 1.
    num_spks: int = 3  # CSS_with_Conformer had 2 spk masks. NOTSOFAR has 3.


@dataclass
class ConformerCssCfg:
    extractor_conf: ExtractorCfg = field(default_factory=ExtractorCfg)
    nnet_conf: NnetCfg = field(default_factory=NnetCfg)


class ConformerCssWrapper(nn.Module):
    """A thin wrapper around the Executor class, built to minimize code changes and to accept CssWithConformerCfg."""
    def __init__(self, cfg: ConformerCssCfg):
        super().__init__()
        nnet = ConformerCSS(**asdict(cfg.nnet_conf))
        self.executor = Executor(nnet, extractor_kwargs=asdict(cfg.extractor_conf), get_mask=True)

    def forward(self, mix: th.Tensor):
        """Compute the masks for the given mixture.

        Args:
            mix (th.Tensor): The mixture of shape [Batch, T, Mics], in time domain, to compute the masks for.
        Returns: A dictionary with these keys:
            'spk_masks' (th.Tensor): The masks for the speakers. Shape: [Batch, F, T, num_spks].
            'noise_masks' (th.Tensor): The masks for the noise. Shape: [Batch, F, T, num_nois].
        """
        assert (mix.shape[2] == 1) == (self.executor.extractor.ipd_extractor is None), \
            (f"IPD extractor is expected iff the number of microphones is greater than 1. "
             f"This may indicate model misconfiguration!")

        if mix.shape[2] == 1:
            mix = mix.squeeze(2)  # [Batch, T, 1] -> [Batch, T]
        else:
            mix = mix.moveaxis(1, 2).contiguous()  # [Batch, T, Mics] -> [Batch, Mics, T]

        res = self.executor({"mix": mix})

        all_masks = th.cat([m.unsqueeze(-1) for m in res], dim=-1)

        assert all_masks.shape[-1] == self.executor.nnet.num_spks + self.executor.nnet.num_nois, \
            f"Expected {self.executor.nnet.num_spks + self.executor.nnet.num_nois} masks, got {all_masks.shape[-1]}!"

        return {
            'spk_masks': all_masks[..., :self.executor.nnet.num_spks],
            'noise_masks': all_masks[..., self.executor.nnet.num_spks:]
        }

    def stft(self, s: th.Tensor):
        """Compute the STFT of a signal.

        Args:
            s (th.Tensor): The signal to compute the STFT of. Shape: [Batch, T, Mics] for multi-channel, [Batch, T] for
                single-channel, all in time domain.

        Returns:
            A tuple of magnitude and phase tensors, both of shape [Batch, F, T, Mics].
        """

        if s.dim() == 3:
            s = s.moveaxis(1, 2).contiguous()  # [Batch, T, Mics] -> [Batch, Mics, T], and make contiguous.

        res = self.executor.extractor.stft(s, cplx=False)  # -> (mag, phase) tuples of [Batch, Mics, F, T]

        if s.dim() == 3:
            res = tuple(r.moveaxis(1, 3).contiguous() for r in res)  # -> (mag, phase) tuples of [Batch, F, T, Mics]

        return res


# TODO: Remove before release.
class DummyCss(nn.Module):
    """A dummy CSS model that does nothing."""
    def __init__(self):
        super().__init__()

        l = nn.Linear(4096, 4096)
        layers = []
        for i in range(5000):
            layers.append(l)
            layers.append(nn.ReLU())

        self.seq = nn.Sequential(*layers)


    def forward(self, mix: th.Tensor):

        # Flatten the mix, except the batch dimension.
        mix = mix.flatten(start_dim=1)
        # Take the first items to match the expected input size of the linear1 layer.
        mix = mix[:, :4096]
        # Pass the mix through the linear layers, with relu in between.
        mix = self.seq(mix)

        return {
            'spk_masks': mix,
            'noise_masks': mix+1
        }