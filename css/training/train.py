import argparse
import dataclasses
import shutil
from itertools import takewhile, count
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import random
from pathlib import Path
from typing import Optional, Tuple
from pprint import pprint

import torch
import torch.distributed as dist
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler

import css.training.losses
from css.training.augmentations import MicShiftAugmentation
from css.training.schedulers import LinearWarmupDecayCfg, LinearWarmupDecayScheduler
from css.training.simulated_dataset import SegmentSplitter, SimulatedDataset
from css.training.conformer_wrapper import ConformerCssCfg, ConformerCssWrapper, DummyCss
from utils.torch_utils import get_world_size, get_rank, is_dist_initialized, move_to, catch_unused_params, \
    is_zero_rank, reduce_dict_to_rank0, is_dist_env_available
import utils.conf
import numpy as np


# Enable cuDNN to dynamically find the most efficient algorithms for the current configuration.
# Determined to be crucial for performance based on our observations.
torch.backends.cudnn.benchmark = True


@dataclass
class SimulatedDatasetCfg:
    sample_frac: float = 1.0
    max_urls: Optional[int] = None


@dataclass
class SchedulerStepLrCfg:
    step_size: int = 1
    gamma: float = 1.0  # default is no decay


@dataclass
class TrainCfg:
    train_dir: str
    val_dir: str
    out_dir: str

    train_set_cfg: SimulatedDatasetCfg = field(default_factory=SimulatedDatasetCfg)
    val_set_cfg: SimulatedDatasetCfg = field(default_factory=SimulatedDatasetCfg)

    single_channel: bool = False

    segment_len_secs: float = 3.0
    fs: int = 16000
    segment_min_overlap_factor: float = 1/6
    segment_max_overlap_factor: float = 1/2
    segment_pr_force_align: float = 0.5

    learning_rate: float = 1e-3
    global_batch_size: int = 32  # global means across all GPUs, local means per GPU
    clip_grad_norm: float = 0.01
    # clips the ground truth to the mixture to avoid trying to drive the mask above 1. "True" is recommended.
    clip_gt_to_mixture: bool = False
    weight_decay: float = 1e-4
    is_debug: bool = False  # no data workers, no DataParallel, etc.
    log_params_mlflow: bool = True
    log_metrics_mlflow: bool = True
    seed: int = 59438191
    dataloader_workers: int = 8  # num_workers for all data loaders

    model_name: str = 'css_with_conformer'
    conformer_css_cfg: ConformerCssCfg = field(default_factory=ConformerCssCfg)

    scheduler_name: str = 'step_lr'  # Options: {'step_lr', 'linear_warmup_decay'}
    scheduler_step_lr_cfg: SchedulerStepLrCfg = field(default_factory=SchedulerStepLrCfg)
    scheduler_linear_warmup_decay_cfg: LinearWarmupDecayCfg = field(default_factory=LinearWarmupDecayCfg)

    # Specify when to evaluate the model, save it, update the scheduler, and stop training. The format is (N,
    # 'epochs'/'iterations') or None for never.
    eval_every: Optional[Tuple] = (1, 'epochs')
    save_every: Optional[Tuple] = None
    scheduler_step_every: Optional[Tuple] = (1, 'epochs')
    stop_after: Optional[Tuple] = (120, 'epochs')
    calc_side_info: bool = False
    loss_name: Optional[str] = None
    base_loss_name: Optional[str] = None


def get_model(cfg: TrainCfg):
    if cfg.model_name == 'css_with_conformer':
        return ConformerCssWrapper(cfg.conformer_css_cfg)
    else:
        raise ValueError(f'Unknown model name: {cfg.model_name}!')


def run_training_css(train_cfg: TrainCfg, train_dir, val_dir, out_dir) -> str:
    assert torch.backends.cudnn.benchmark

    # Initialize mlflow if available
    attempt_load_mlflow()

    # Print (and possibly log with mlflow) the config
    log('Starting training')
    pprint(train_cfg)

    if train_cfg.log_params_mlflow:
        log_params_to_mlflow(train_cfg)

    # Determine if we are running in distributed mode and initialize accordingly
    if is_dist_env_available():
        dist.init_process_group('nccl')

        log(f'Distributed: {get_rank()=}, {get_world_size()=}')

        # NOTE! must call set_device or allocations go to GPU 0 disproportionally, causing CUDA OOM.
        torch.cuda.set_device(get_rank())
        torch.distributed.barrier()
        log(f'---------init_process_group() Done')

    # Determine the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not is_dist_initialized() \
        else torch.device(f'cuda:{get_rank()}')

    print(f'Device: {device}')

    # Seed weight initialization etc.
    random.seed(train_cfg.seed + 1257)
    os.environ['PYTHONHASHSEED'] = str(train_cfg.seed + 1417)
    np.random.seed(train_cfg.seed + 4179)
    torch.manual_seed(train_cfg.seed + 40973)
    torch.cuda.manual_seed(train_cfg.seed + 425291)

    # Instantiate the model
    model = get_model(train_cfg)

    # Move the model to the device
    model = model.to(device)

    # Wrap the model in DDP or DP if needed
    if is_dist_initialized():
        # DDP
        # input tensors are assumed to be batch-first
        model = nn.parallel.DistributedDataParallel(model, dim=0,
                                                    device_ids=[get_rank()],
                                                    output_device=get_rank())
        dist.barrier()
    elif not device.type == 'cpu' and not train_cfg.is_debug:
        # DP
        log(f'Using {torch.cuda.device_count()} GPU(s)')
        model = nn.DataParallel(model, dim=0)

    # TODO support various optimizers?
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )

    # Set the model to training mode
    model.train()

    # Set up a scheduler
    if train_cfg.scheduler_name == 'step_lr':
        scheduler = StepLR(optimizer, **asdict(train_cfg.scheduler_step_lr_cfg))
    elif train_cfg.scheduler_name == 'linear_warmup_decay':
        scheduler = LinearWarmupDecayScheduler(optimizer, train_cfg.scheduler_linear_warmup_decay_cfg)
    else:
        raise ValueError(f'Unknown scheduler name: {train_cfg.scheduler_name}!')

    # PIT loss
    base_loss_fn = css.training.losses.mse_loss
    pit_loss = css.training.losses.PitWrapper(base_loss_fn)

    # Construct the datasets
    seg_len_samples = train_cfg.segment_len_secs * train_cfg.fs
    seg_split = SegmentSplitter(
        min_overlap=int(seg_len_samples * train_cfg.segment_min_overlap_factor),
        max_overlap=int(seg_len_samples * train_cfg.segment_max_overlap_factor),
        pr_force_align=train_cfg.segment_pr_force_align,
        desired_segm_len=int(seg_len_samples))

    needed_columns = ['mixture', 'gt_spk_direct_early_echoes', 'gt_noise']
    train_set = SimulatedDataset(dataset_path=train_dir, segment_split_func=seg_split,
                                 seed=44697134, **asdict(train_cfg.train_set_cfg),
                                 single_channel=train_cfg.single_channel, needed_columns=needed_columns)
    log(f'Training set stats: {len(train_set)} segments, {train_set.get_length_seconds() / 3600:.4} hours')
    # train_set = DummySimulatedDataset(num_samples=10000000)
    # log('Using dummy training set!')

    val_set = SimulatedDataset(dataset_path=val_dir, segment_split_func=seg_split,
                               seed=836591172, **asdict(train_cfg.val_set_cfg),
                               single_channel=train_cfg.single_channel, needed_columns=needed_columns)
    log(f'Validation set stats: {len(val_set)} segments, {val_set.get_length_seconds() / 3600:.4} hours')
    # val_set = DummySimulatedDataset(num_samples=100)
    # log('Using dummy validation set!')

    # Data augmentation
    # TODO: Document the rationale for the augmentation on the GPU.
    augmentation_fns = (MicShiftAugmentation(seed=train_cfg.seed + 2112, device=device),) \
        if not train_cfg.single_channel else ()

    # samplers
    if is_dist_initialized():
        # DistributedSampler seeds index permutation with seed+epoch. Must call set_epoch() before iter()!
        sampler_train = DistributedSampler(train_set, shuffle=True, seed=train_cfg.seed + 46117)
        sampler_val = DistributedSampler(val_set, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)

    # Instantiate the data loaders
    num_workers = 0 if train_cfg.is_debug else train_cfg.dataloader_workers
    local_batch_size = train_cfg.global_batch_size // get_world_size()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=sampler_train,
        batch_size=local_batch_size,
        shuffle=False,  # already done by sampler
        num_workers=num_workers,
        pin_memory=True
        # TODO: is default collate_fn good?
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        sampler=sampler_val,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Calculate the total number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Model size: {total_params:,} params')

    # Calculate the number of epochs to train for
    assert train_cfg.stop_after is not None, '`stop_after` must be specified!'
    assert train_cfg.stop_after[1] in ('epochs', 'iterations'), \
        f'Unknown frequency type specified for `stop_after`: {train_cfg.stop_after[1]}!'
    expected_num_epochs = train_cfg.stop_after[0] / len(train_loader) if train_cfg.stop_after[1] == 'iterations' \
        else train_cfg.stop_after[0]
    log(f'Expected number of epochs to train for: {expected_num_epochs:.2f}')

    # Report some stats to mlflow
    if train_cfg.log_metrics_mlflow and is_zero_rank():
        def set_metrics(dataset, loader, prefix):
            m = {'num_samples': len(dataset), 'num_hours': dataset.get_length_seconds() / 3600, 'num_batches': len(loader)}
            m = {prefix + k: v for k, v in m.items()}
            return m

        log_metrics_to_mlflow({
            **set_metrics(train_set, train_loader, 'train/'),
            **set_metrics(val_set, val_loader, 'val/'),
            'model_total_params': total_params,
            'expected_num_epochs': expected_num_epochs,
            'local_batch_size': local_batch_size,
        }, step=0)

    # Set up some variables for training
    loss_sum = 0.0
    num_instances = 0
    total_iters = 1  # Note that iterations and epochs are 1-based
    stop = False

    # Training loop
    for epoch in takewhile(lambda _: not stop, count(start=1)):
        log(f'Starting epoch {epoch}')
        if is_dist_initialized() and hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)

        num_batches = len(train_loader)

        for iter_in_epoch, batch in takewhile(lambda _: not stop, enumerate(train_loader, start=1)):

            def loop_log(*args):
                log(f'ep{epoch} it{iter_in_epoch}/{num_batches} tot_it{total_iters}:', *args)

            # Print progress
            if iter_in_epoch % 10 == 0:
                loop_log('.')

            # Move to device
            batch = move_to(batch, device)

            # Augment
            for aug_fn in augmentation_fns:
                batch = aug_fn(batch)

            # Verify that the model is in training mode
            assert model.training, 'Model is not in training mode!'

            # Calculate loss
            loss = _calc_loss(batch, model, base_loss_fn, pit_loss, clip_gt_to_mixture=train_cfg.clip_gt_to_mixture)

            # with DataParallel, loss is either [Batch] or [#GPU] tensor depending on model

            # Zero the gradients and perform a backward pass
            optimizer.zero_grad()
            assert not loss.isnan()
            loss.backward()

            if train_cfg.is_debug:
                catch_unused_params(model)

            # Clip the gradients
            if train_cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)

            # Update the parameters
            optimizer.step()

            # Collect the running loss
            current_bs = batch['mixture'].shape[0]
            loss_sum += current_bs * loss.data.item()
            num_instances += current_bs

            def is_every(freq):
                if freq is None:
                    return False
                elif freq[1] == 'epochs':
                    return epoch % freq[0] == 0 and iter_in_epoch == num_batches  # iter_in_epoch is 1-based
                elif freq[1] == 'iterations':
                    return total_iters % freq[0] == 0
                else:
                    raise ValueError(f'Unknown frequency type: {freq[1]}!')

            # Update the scheduler
            if is_every(train_cfg.scheduler_step_every):
                scheduler.step()

            # Stop training
            if is_every(train_cfg.stop_after):
                loop_log('Stopping training')
                stop = True
                # Note that we don't break here to allow the model to be evaluated and saved below.

            # Evaluate the model
            if is_every(train_cfg.eval_every) or stop:
                loop_log('Evaluating')
                torch.cuda.empty_cache()

                # Get the worker-local metrics for train/val sets
                train_metrics = {'loss': loss_sum, 'num_instances': num_instances}
                val_metrics = eval_model(model, val_loader, device, base_loss_fn, pit_loss,
                                         clip_gt_to_mixture=train_cfg.clip_gt_to_mixture)

                # Reduce the metrics to rank0
                train_metrics = reduce_metrics_to_rank0(train_metrics, device, prefix='train/')
                val_metrics = reduce_metrics_to_rank0(val_metrics, device, prefix='val/')

                if is_zero_rank():
                    # Other metrics
                    lr = scheduler.get_last_lr()
                    assert len(lr) == 1
                    other_metrics = {'lr': lr[0]}

                    # Combine metrics
                    all_metrics = {**train_metrics, **val_metrics, **other_metrics}

                    # Log the metrics
                    loop_log(all_metrics)
                    if train_cfg.log_metrics_mlflow:  # will be done on rank0 only (see outer if)
                        log_metrics_to_mlflow(
                            all_metrics,
                            # Note that the step depends on the frequency type
                            step=epoch if train_cfg.eval_every[1] == 'epoch' else total_iters)

                # Reset the running loss
                loss_sum = 0.0
                num_instances = 0

            # Save the model
            if is_every(train_cfg.save_every) or stop:

                model_path = f'/model_epoch_{epoch:04}.pt' \
                    if train_cfg.save_every is not None and train_cfg.save_every[1] == 'epoch' else \
                    f'/model_iter_{total_iters:07}.pt'
                model_path = out_dir + model_path

                if is_zero_rank():
                    loop_log(f'Saving to {model_path}.')

                    os.makedirs(out_dir, exist_ok=True)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, model_path)

            # Increment the total iterations counter
            total_iters += 1

    log(f'Finished training')

    if dist.is_initialized():
        dist.destroy_process_group()

    return out_dir


def _calc_loss(segment_batch, model, base_loss_fn, pit_loss, clip_gt_to_mixture: bool):

    ref_mic = 0  # reference microphone index

    # Perform a forward pass
    mix = segment_batch['mixture']  # -> [Batch, T, Mics] (time domain signal)
    res = model(mix)  # -> # (spks, noise) tuple of [Batch, F, T, #spks/noise] tensors (freq domain)

    module = getattr(model, 'module', model)  # handle DP/DDP module attribute
    # Apply STFT on the mic0 mixture, take the magnitude, and add a singleton dimension to allow broadcasting
    # over speakers/noise.
    mix_mic0_mag_ft = module.stft(mix[:, :, ref_mic]).abs()[..., None]  # -> magnitude [Batch, F, T, 1]

    # Calculate speakers and noise magnitude spectrograms after masking
    pred_spks = res['spk_masks'] * mix_mic0_mag_ft  # -> [Batch, F, T, #spks]
    pred_noise = res['noise_masks'] * mix_mic0_mag_ft  # -> [Batch, F, T, #noise]
    assert pred_noise.shape[-1] == 1, f'NOTSOFAR supports one noise mask, got {pred_noise.shape[-1]}!'
    pred_noise = pred_noise.squeeze(-1)  # -> [Batch, F, T]

    # Apply STFT on the ground truth signals and take the magnitude
    gt_spks = _get_gt_mic0_stft_mag(segment_batch['gt_spk_direct_early_echoes'], model, ref_mic)  # -> magnitude [Batch, F, T, #spks]
    gt_noise = module.stft(segment_batch['gt_noise'][:, :, ref_mic]).abs() # -> magnitude [Batch, F, T]

    # Clip the ground truth to the mixture to avoid trying to drive the mask above 1. Context: The CSS with Conformer
    # model applies a sigmoid activation at the top.
    if clip_gt_to_mixture:
        gt_spks = torch.min(gt_spks, mix_mic0_mag_ft)
        assert mix_mic0_mag_ft.shape[-1] == 1  # the last singleton dimension was added above, sanity check
        gt_noise = torch.min(gt_noise, mix_mic0_mag_ft[..., 0])

    # Calculate the loss
    noise_loss = base_loss_fn(pred_noise, gt_noise).mean(dim=(1, 2))  # -> [Batch]
    spk_loss, target_perm = pit_loss(pred_spks, gt_spks)  # -> [Batch]
    loss = spk_loss + noise_loss  # -> [Batch]

    return loss.mean()


def _get_gt_mic0_stft_mag(gt, model, ref_mic: int):
    # Fetch mic0
    gt_mic0 = gt[:, :, ref_mic, :]  # [Batch, T, Mics, Max_spks] -> [Batch, T, Max_spks]
    max_spks = gt_mic0.shape[-1]

    module = getattr(model, 'module', model)  # handle DP/DDP module attribute

    # Collate the batch and max_spks dimensions
    gt_mic0 = gt_mic0.moveaxis(-1, 1).contiguous()  # -> [Batch, Max_spks, T]
    gt_mic0 = gt_mic0.view(-1, gt_mic0.shape[2])  # -> [Batch*Max_spks, T]
    gt_cplx_ft = module.stft(gt_mic0)  # -> complex [Batch*Max_spks, F, T]

    # Undo the collation. Work on the magnitude spectrograms only.
    gt_mag_ft = gt_cplx_ft.abs()
    gt_mag_ft = gt_mag_ft.view(-1, max_spks, * gt_mag_ft.shape[1:])  # -> [Batch, Max_spks, F, T]
    gt_mag_ft = gt_mag_ft.moveaxis(1, -1).contiguous()  # -> [Batch, F, T, Max_spks]
    return gt_mag_ft


@torch.no_grad()
def eval_model(model, dataloader, device, base_loss_fn, pit_loss, clip_gt_to_mixture):
    """ Evaluate model. DistributedDataParallel (DDP) supported. """

    # Record the model's current mode and set it to eval
    was_training = model.training
    model.eval()

    try:
        loss_sum = 0.0
        num_instances = 0

        num_batches = len(dataloader)
        for it, batch in enumerate(dataloader):

            # Print progress
            if it % 10 == 0:
                log(f'eval it{it}/{num_batches}')

            # Move to device
            batch = move_to(batch, device)

            # Calculate loss
            loss = _calc_loss(batch, model, base_loss_fn, pit_loss, clip_gt_to_mixture=clip_gt_to_mixture)

            current_bs = batch['mixture'].shape[0]
            loss_sum += current_bs * loss.data.item()
            num_instances += current_bs

        # Note that sums should be returned here, not averages. The averages will be calculated in
        # reduce_metrics_to_rank0().
        return {'loss': loss_sum, 'num_instances': num_instances}

    finally:
        # Restore the model's mode
        model.train(was_training)


def reduce_metrics_to_rank0(worker_metrics, device, prefix = None):

    def calc_average_metrics(metrics):
        res = {k: v / metrics['num_instances'] for k, v in metrics.items()
               if k != 'num_instances'}
        res['num_instances'] = metrics['num_instances']
        return res

    if is_dist_initialized():
        # To device
        worker_metrics = {k: torch.tensor(v).to(device) for k, v in worker_metrics.items()}

        # Sum all processes into rank0
        reduced_metrics = reduce_dict_to_rank0(worker_metrics, average=False)

        if not is_zero_rank():
            return None

        # Calculate the average metrics
        avg_metrics = calc_average_metrics(reduced_metrics)

        # Back to cpu
        avg_metrics = {k: v.cpu().item() for k, v in avg_metrics.items()}

    else:
        avg_metrics = calc_average_metrics(worker_metrics)

    # Prefix the metrics if needed
    if prefix is not None:
        avg_metrics = {prefix + k: v for k, v in avg_metrics.items()}

    return avg_metrics


# Mlflow-related functions
mlflow = None


def attempt_load_mlflow():
    global mlflow
    if mlflow is None:
        try:
            mlflow = __import__('mlflow')
        except ImportError:
            pass


def log_params_to_mlflow(d, prefix=''):
    if mlflow is None:
        return

    for f in dataclasses.fields(d):
        field_value = getattr(d, f.name)

        full_field_name = f'{prefix}{f.name}'

        if dataclasses.is_dataclass(field_value):
            log_params_to_mlflow(field_value, prefix=full_field_name + '.')
        else:
            # Note that complex types (e.g. dict) will be logged as strings
            # TODO: Consider rewriting this function to use mlflow.log_params() to possibly make it more efficient.
            mlflow.log_param(full_field_name, field_value)


def log_metrics_to_mlflow(metrics: dict, step: int):
    if mlflow is None:
        return

    mlflow.log_metrics(metrics, step=step)


def log(*args, **kwargs):
    """Prints a nicely formatted log line."""

    print(datetime.now().strftime('%d/%m/%Y %H:%M:%S'), flush=True, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default=None)
    parser.add_argument('--data_root_in', default=None)
    parser.add_argument('--data_root_out', default=None)
    args = parser.parse_args()

    # debug_mc.yaml:
    # 1. Sets is_debug=True, which turns off data workers, DataParallel etc. to ease debugging.
    # 2. Uses the tiny sample_data/css_train_set as train and validation sets.

    # Take care of paths
    project_dir = Path(__file__).parents[2]
    conf_path = str(project_dir / 'configs' / 'train_css' / 'local' / 'debug_mc.yaml') if args.conf is None \
        else args.conf
    data_root_in = project_dir if args.data_root_in is None \
        else Path(args.data_root_in)
    data_root_out = project_dir / 'artifacts' / 'outputs' / 'css_train' if args.data_root_out is None \
        else Path(args.data_root_out)

    # Load the config
    train_cfg = utils.conf.load_yaml_to_dataclass(str(conf_path), TrainCfg)

    # Append the paths specified in the config to the roots above
    train_dir = data_root_in / train_cfg.train_dir
    val_dir = data_root_in / train_cfg.val_dir
    out_dir = data_root_out / train_cfg.out_dir

    # Copy the config to the output directory with a fixed name
    log(f'Copying the config to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(conf_path, str(out_dir / 'config.yaml'))

    # Run training
    run_training_css(train_cfg, train_dir=str(train_dir), val_dir=str(val_dir), out_dir=str(out_dir))

    # Once training is done, you can plug the checkpoint and yaml files into css_inference()
    # (see load_separator_model() in css/css.py)


if __name__ == '__main__':
    main()
