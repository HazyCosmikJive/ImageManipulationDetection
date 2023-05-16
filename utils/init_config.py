import os
import yaml
from easydict import EasyDict as edict
import datetime

def init_config(arg):
    assert os.path.exists(arg.config), "Config {} do not exist".format(arg.config)
    with open(arg.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config["config"])

    if arg.debug:
        config.common.debug = True
    
    config.common.timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    config.common.checkpointpath = os.path.join("./work_dirs", config.common.exp_tag, "checkpoints")
    config.common.logpath = os.path.join("./work_dirs", config.common.exp_tag, "logs")
    config.common.tblogpath = os.path.join("./work_dirs", config.common.exp_tag, "tensorlogs")
    config.common.predpath = os.path.join("./work_dirs", config.common.exp_tag, "preds")
    os.makedirs(config.common.checkpointpath, exist_ok=True)
    os.makedirs(config.common.logpath, exist_ok=True)
    os.makedirs(config.common.tblogpath, exist_ok=True)
    os.makedirs(config.common.predpath, exist_ok=True)
    config.common.workspace = os.path.join("./work_dirs", config.common.exp_tag)
    if config.common.get("debug", False):
        config.common.checkpointpath_debug = os.path.join("./work_dirs", config.common.exp_tag, "debug/checkpoints")
        config.common.logpath_debug = os.path.join("./work_dirs", config.common.exp_tag, "debug/logs")
        config.common.tblogpath_debug = os.path.join("./work_dirs", config.common.exp_tag, "debug/tensorlogs")
        os.makedirs(config.common.checkpointpath_debug, exist_ok=True)
        os.makedirs(config.common.logpath_debug, exist_ok=True)
        os.makedirs(config.common.tblogpath_debug, exist_ok=True)
        config.common.workspace = os.path.join("./work_dirs", config.common.exp_tag, "debug")

    return config
