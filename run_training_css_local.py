from pathlib import Path

import utils.conf
from css.training.train import run_training_css, TrainCfg

if __name__ == '__main__':

    project_dir = Path(__file__).parent

    # debug_mc.yaml:
    # 1. Sets is_debug=True, which turns off data workers, DataParallel etc. to ease debugging.
    # 2. Uses the tiny sample_data/css_train_set as train and validation sets.
    conf_path = str(project_dir / 'configs' / 'train_css' / 'local' / 'debug_mc.yaml')
    data_root_in = project_dir
    # checkpoint will be written here
    data_root_out = project_dir / 'artifacts' / 'outputs' / 'css_train'

    # Load the config.
    train_cfg = utils.conf.get_conf(str(conf_path), TrainCfg)

    # Run training
    run_training_css(train_cfg, data_root_in, data_root_out)

    # Once training is done, you can plug the checkpoint and yaml files into css_inference()
    # (see load_separator_model() in css/css.py)