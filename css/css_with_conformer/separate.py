#!/usr/bin/env python
"""

This module and the contents of the "css_with_conformer" folder were adapted from
https://github.com/Sanyuan-Chen/CSS_with_Conformer, with some modifications.

"""

import yaml
import argparse
from pathlib import Path

import torch as th
import numpy as np
import soundfile as sf

from css.css_with_conformer.nnet import supported_nnet
from css.css_with_conformer.executor.executor import Executor
from css.css_with_conformer.utils.audio_util import WaveReader
from css.css_with_conformer.utils.mvdr_util import make_mvdr


class EgsReader(object):
    """
    Egs reader
    """
    def __init__(self,
                 mix_scp,
                 sr=16000):
        self.mix_reader = WaveReader(mix_scp, sr=sr)

    def __len__(self):
        return len(self.mix_reader)

    def __iter__(self):
        for key, mix in self.mix_reader:
            egs = dict()
            egs["mix"] = mix
            yield key, egs


class Separator(object):
    """
    A simple wrapper for speech separation
    """
    def __init__(self, cpt_dir, get_mask=False, device='cpu'):
        # load executor
        cpt_dir = Path(cpt_dir)
        self.get_mask = get_mask
        self.executor = self._load_executor(cpt_dir)
        cpt_ptr = cpt_dir / "best.pt.tar"
        epoch = self.executor.resume(cpt_ptr.as_posix())
        print(f"Load checkpoint at {cpt_dir}, on epoch {epoch}")
        #print(f"Nnet summary: {self.executor}")
        self.device = device
        self.executor.to(self.device)
        self.executor.eval()

    def separate(self, egs):
        """
        Do separation
        """
        egs["mix"] = th.from_numpy(egs["mix"][None, :]).to(self.device, non_blocking=True)
        with th.no_grad():
            spks = self.executor(egs)
            spks = [s.detach().squeeze().cpu().numpy() for s in spks]
            return spks

    def _load_executor(self, cpt_dir):
        """
        Load executor from checkpoint
        """
        with open(cpt_dir / "train.yaml", "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        nnet_type = conf["nnet_type"]
        if nnet_type not in supported_nnet:
            raise RuntimeError(f"Unknown network type: {nnet_type}")
        nnet = supported_nnet[nnet_type](**conf["nnet_conf"])
        executor = Executor(nnet, extractor_kwargs=conf["extractor_conf"], get_mask=self.get_mask)
        return executor


def run(args):
    wav, sr = sf.read(args.wav_file, dtype='float32')

    # separator
    seperator = Separator(args.checkpoint, device_id=args.device_id, get_mask=args.mvdr)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)
    egs = {'mix': wav[int(sr * args.start): int(sr * args.end)]}
    duration_sec = egs['mix'].size / sr

    # print(f"Start Separation " + ("w/ mvdr" if args.mvdr else "w/o mvdr"))
    # for key, egs in egs_reader:
    # print(f"Processing utterance {key}...{egs}")
    mixed = egs["mix"]
    print('mixed',mixed.shape)
    spks = seperator.separate(egs)
    print('spks',len(spks),spks[0].shape)

    if args.mvdr:
        res1, res2 = make_mvdr(spks[:2], spks[2:], np.asfortranarray(mixed.T))
        spks = [res1, res2]

    sf.write(dump_dir / f"{duration_sec}_mix.wav", egs['mix'][0].cpu().numpy(), sr)
    for i, s in enumerate(spks):
        if i < args.num_spks:
            write_path = dump_dir / f"{duration_sec}_{i}.wav"
            print(write_path)
            sf.write(write_path, s * 0.9 / np.max(np.abs(s)), sr)

    print(f"Done processing {args.wav_file}")


def run_pretrained_sc_conformer():
    args = argparse.Namespace(
        checkpoint=r"C:\Repos\NOTSOFAR\artifacts\css_models\CSS_with_Conformer\sc\1ch_conformer_base",
        wav_file=r"C:\Repos\NOTSOFAR\artifacts\ch0.wav",
        start=28,
        end=32,
        num_spks=2,
        device_id=0,
        sr=16000,
        dump_dir=r"C:\Repos\NOTSOFAR\artifacts\conformer_dump",
        mvdr=False
    )

    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do speech separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument("--wav-file",
                        type=str,
                        required=True,
                        help="mixed audio wav")
    parser.add_argument("--start",
                        type=float,
                        required=False,
                        default=None,
                        help="audio processing start (secs)")
    parser.add_argument("--end",
                        type=float,
                        required=False,
                        default=None,
                        help="audio processing end (secs)")
    parser.add_argument("--num_spks",
                        type=int,
                        default=2,
                        help="Number of the speakers")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, -1 means "
                        "running on CPU")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate for mixture input")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="sep",
                        help="Directory to dump separated speakers")
    parser.add_argument("--mvdr",
                        type=bool,
                        default=False,
                        help="apply mvdr")
    args = parser.parse_args()

    run(args)
