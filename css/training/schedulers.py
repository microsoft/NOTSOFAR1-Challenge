import warnings
from dataclasses import dataclass
import torch.optim.lr_scheduler


@dataclass
class LinearWarmupDecayCfg:
    # Defaults are set according to the CSS with Conformer paper
    warmup: int = 10000
    decay: int = 260000


class LinearWarmupDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, cfg: LinearWarmupDecayCfg, verbose=False):
        self.cfg = cfg
        super().__init__(optimizer, self._lr_lambda, verbose=verbose)

    def _lr_lambda(self, step):
        if step < self.cfg.warmup:
            res = step / self.cfg.warmup
        elif step < self.cfg.warmup + self.cfg.decay:
            res = 1 - (step - self.cfg.warmup) / self.cfg.decay
        else:
            if step > self.cfg.warmup + self.cfg.decay:
                warnings.warn(f'Learning rate has been decayed to zero! {step=}')
            res = 0

        if self.verbose:
            print(f'LinearWarmupDecayScheduler: {step=} {res=}')

        return res
