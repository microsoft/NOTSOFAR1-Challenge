import argparse
from pathlib import Path
from pprint import pprint
from typing import Literal

from inference_pipeline.inference import InferenceCfg, inference_pipeline, FetchFromCacheCfg
from utils.azure_storage import download_meeting_subset, download_models
from utils.conf import load_yaml_to_dataclass, update_dataclass

ConfigName = Literal['full_dev_set_mc', 'full_dev_set_sc', 'dev_set_1_mc_debug']


def get_project_root() -> Path:
    """ Returns project root folder """
    return Path(__file__).parent


def load_config(config_name: ConfigName) -> InferenceCfg:
    """ Returns the config file path and session query for the given config name """
    project_root = get_project_root()

    updates = {}
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

    cfg: InferenceCfg = load_yaml_to_dataclass(str(conf_file), InferenceCfg)
    cfg = update_dataclass(cfg, updates)

    if session_query is not None:
        assert cfg.session_query is None, 'overriding session_query from yaml'
        cfg.session_query = session_query

    return cfg


def main(config_name: ConfigName = 'dev_set_1_mc_debug', output_dir: str = ""):
    project_root = get_project_root()
    cfg: InferenceCfg = load_config(config_name)

    # download the entire dev-set (all sessions, multi-channel and single-channel)
    meetings_root = project_root / 'artifacts' / 'meeting_data'
    dev_meetings_dir = download_meeting_subset(subset_name='dev_set',  # dev-set is without GT for now
                                               version='240208.2_dev',
                                               destination_dir=str(meetings_root))

    if dev_meetings_dir is None:
        raise RuntimeError('failed to download benchmark dataset')

    # download models
    models_dir = project_root / 'artifacts' / 'css_models'
    download_models(destination_dir=str(models_dir))

    # outputs per module will be written here
    outputs_dir = (project_root if output_dir == "" else Path(output_dir)) / 'artifacts' / 'outputs'

    cache_cfg = FetchFromCacheCfg()  # no cache, use this at your own risk.

    exp_name = ('pass_through' if cfg.css.pass_through_ch0 else 'css') + '_' + cfg.asr.model_name
    outputs_dir = outputs_dir / exp_name
    
    pprint(f'{config_name=}')
    pprint(cfg)

    # run inference pipeline
    inference_pipeline(meetings_dir=str(dev_meetings_dir),
                       models_dir=str(models_dir),
                       out_dir=str(outputs_dir),
                       cfg=cfg,
                       cache=cache_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference pipeline')
    parser.add_argument('--config-name', type=str, default="dev_set_1_mc_debug",
                        help='Config scenario for the inference, default: dev_set_1_mc_debug')
    parser.add_argument('--output-dir', type=str, default="",
                        help='Output directory path, default: ./artifacts/outputs')
    args = parser.parse_args()

    main(args.config_name, args.output_dir)
