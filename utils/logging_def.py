import logging

import pandas as pd

# this must be called before any other loggers are instantiated to take effect
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(name)s]  %(message)s')

# verbose pandas display
pd.set_option('display.precision', 4)
pd.options.display.width = 600
pd.options.display.max_columns = 20
pd.options.display.max_rows = 200
# _LOG.info('display options:\n%s', pprint.pformat(pd.options.display.__dict__, indent=4))


def get_logger(name: str):
    """
    All modules should use this function to get a logger.
    This way, we ensure all loggers are instantiated after basicConfig() call and inherit the same config.
    """
    return logging.getLogger(name)