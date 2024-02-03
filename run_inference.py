from pprint import pprint
from pathlib import Path
from typing import Literal

from inference_pipeline.inference import InferenceCfg, inference_pipeline, FetchFromCacheCfg
from utils.azure_storage import download_meeting_subset, download_models
from utils.conf import get_conf

if __name__ == "__main__":
    project_root = Path(__file__).parent

    config_name: Literal['full_dev_set_mc', 'full_dev_set_sc', 'dev_set_1_mc_debug'] = 'dev_set_1_mc_debug'

    if config_name == 'full_dev_set_mc':
        # all multi-channel (MC) dev-set sessions
        conf_file = project_root / 'configs/inference/inference_v1.yaml'
        session_query = "is_mc == True"  # filter only MC

    elif config_name == 'full_dev_set_sc':
        # all single-channel (SC) dev-set sessions
        conf_file = project_root / 'configs/inference/inference_v1.yaml'
        session_query = "is_mc == False"  # filter only SC

    elif config_name == 'dev_set_1_mc_debug':
        # for quick debug: 'tiny' Whisper, one MC (multi-channel) session
        conf_file = project_root / 'configs/inference/debug_inference.yaml'
        session_query = 'device_name == "plaza_0" and is_mc == True and meeting_id == "MTG_30860"'
    else:
        raise ValueError(f'unknown config name: {config_name}')

    cfg: InferenceCfg = get_conf(str(conf_file), InferenceCfg)
    if session_query is not None:
        assert cfg.session_query is None, 'overriding session_query from yaml'
        cfg.session_query = session_query

    # download the entire dev-set (all sessions, multi-channel and single-channel)
    meetings_root = project_root / 'artifacts' / 'meeting_data'
    dev_meetings_dir = download_meeting_subset(subset_name='dev_set',  # dev-set is without GT for now
                                               version='240130.1_dev',
                                               destination_dir=str(meetings_root))

    if dev_meetings_dir is None:
        raise RuntimeError('failed to download benchmark dataset')

    # download models
    models_dir = project_root / 'artifacts' / 'css_models'
    download_models(destination_dir=str(models_dir))

    # outputs per module will be written here
    outputs_dir = project_root / 'artifacts' / 'outputs'

    cache_cfg = FetchFromCacheCfg()  # no cache, use this at your own risk.

    pprint(f'{config_name=}')
    pprint(cfg)

    # run inference pipeline
    inference_pipeline(meetings_dir=str(dev_meetings_dir),
                       models_dir=str(models_dir),
                       out_dir=str(outputs_dir),
                       cfg=cfg, cache=cache_cfg)
