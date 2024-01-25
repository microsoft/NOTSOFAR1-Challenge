import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from inference_pipeline.inference import InferenceCfg, inference_pipeline, FetchFromCacheCfg
from utils.azure_storage import download_meeting_subset, download_models
from utils.conf import get_conf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]  %(message)s')

    # verbose pandas display for debugging
    pd.set_option('display.precision', 4)
    pd.options.display.width = 600
    pd.options.display.max_columns = 20
    pd.options.display.max_rows = 200
    # _LOG.info('display options:\n%s', pprint.pformat(pd.options.display.__dict__, indent=4))
    project_root = Path(__file__).parent


    config_name: Literal['full_dev_set_mc', 'full_dev_set_sc', 'dev_set_1_sc'] = 'full_dev_set_mc'

    if config_name == 'full_dev_set_mc':
        # pass-through CSS, large-v2 Whisper, all multi-channel (MC) dev-set sessions
        conf_file = project_root / 'configs/inference/css_passthrough_ch0.yaml'
        session_query = "is_mc == True"  # filter only MC

    elif config_name == 'full_dev_set_sc':
        # pass-through CSS, large-v2 Whisper, all single-channel (SC) dev-set sessions
        conf_file = project_root / 'configs/inference/css_passthrough_ch0.yaml'
        session_query = "is_mc == False"  # filter only SC

    elif config_name == 'dev_set_1_sc':
        # for quick debug: pass-through CSS, tiny Whisper, one MC (multi-channel) session
        conf_file = project_root / 'configs/inference/css_passthrough_ch0_debug.yaml'
        cfg: InferenceCfg = get_conf(str(conf_file), InferenceCfg)
        session_query = None  # yaml already sets session_query
    else:
        raise ValueError(f'unknown config name: {config_name}')

    cfg: InferenceCfg = get_conf(str(conf_file), InferenceCfg)
    if session_query is not None:
        assert cfg.session_query is None, 'overriding session_query from yaml'
        cfg.session_query = session_query


    # download the entire dev-set (all sessions, multi-channel and single-channel)
    meetings_root = project_root / 'artifacts' / 'meeting_data'
    dev_meetings_dir = download_meeting_subset(subset_name='dev_set',
                                               version='240121_dev',
                                               destination_dir=str(meetings_root))
    if dev_meetings_dir is None:
        raise RuntimeError('failed to download benchmark dataset')

    # download models
    models_dir = project_root / 'artifacts' / 'css_models'
    download_models(destination_dir=str(models_dir))

    # outputs per module will be written here
    outputs_dir = project_root / 'artifacts' / 'outputs'

    cache_cfg = FetchFromCacheCfg()  # no cache, use this at your own risk.

    # run inference pipeline
    inference_pipeline(meetings_dir=str(dev_meetings_dir),
                       models_dir=str(models_dir),
                       out_dir=str(outputs_dir),
                       cfg=cfg, cache=cache_cfg)
