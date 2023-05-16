import os
import pprint
import numpy as np
import random
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from losses import SoftBCEWithLogitsLoss

from utils.log import setup_logger
from utils.scheduler import get_scheduler
from utils.parser import parse_cfg
from utils.writer import init_writer
from utils.init_config import init_config
from models.model_entry import model_entry
from data.dataset_entry import get_train_val_loader
from tools.train import train_model
from tools.inference import inference

import ipdb


# ----------------- TODO LIST ----------------
'''
TODO: aug: augmentation params in config
TODO: support multi loss functions
'''
# --------------------------------------------


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    arg = parse_cfg()
    config = init_config(arg)

    seed_everything(config.common.get("random_seed", 42))

    logger = setup_logger(config)
    logger.info("Config:\n {}".format(pprint.pformat(config)))
    # copy current config to workspace
    configname = config.common.exp_tag + ".yaml"
    shutil.copy(arg.config, os.path.join(config.common.workspace, configname))

    # set device (single GPU now)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert len(arg.gpu) == 1, "Single GPU now"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu[0]) # single gpu here
    device = torch.device("cuda:{}".format(str(arg.gpu[0])))

    # tensorboard
    writer = init_writer(config)
    
    # dataloader
    loader_dict = get_train_val_loader(config, logger)
    train_loader, val_loader = loader_dict["train_loader"], loader_dict["val_loader"]

    # build model
    model = model_entry(config, logger).to(device)

    # optimizer
    if config.trainer.optimizer.type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.trainer.optimizer.base_lr, weight_decay=config.trainer.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError("Not implemented optimizer {}".format(config.trainer.optimizer.type))

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    # loss
    losstype = config.model.loss.get("type", "BCE")
    if losstype == "BCE":
        criterion = SoftBCEWithLogitsLoss(
            smooth_factor=0.1, pos_weight=torch.tensor([5.0]).to(device)
        ).to(device)
    else:
        raise NotImplementedError("Loss {} is not implemented".format(losstype))

    # start training
    if config.common.get("train", True):
        logger.info("\n----------------- START TRAINING -----------------")
        train_model(model, config, device, criterion, optimizer, scheduler, train_loader, val_loader, logger, writer)
    if config.common.get("test", False):
        logger.info("\n----------------- START TESTING -----------------")
        test_f1, test_iou = inference(config, logger, model, device)
        logger.info("\n===> TEST RESULT \n     Test F1 {:.4f} Test IoU {:.4f}".format(
                            test_f1, test_iou
                        ))


if __name__ == "__main__":
    main()