import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

def init_writer(config):
    writer = None
    if config.common.get("use_tensorlog", False):
        if config.common.get("debug", False):  # debug mode
            writer_dir = config.common.tblogpath_debug
        else:
            writer_dir = config.common.tblogpath
        writer = Summary(writer_dir=writer_dir, suffix=config.common.timestamp)
    return writer
class Summary(object):
    def __init__(self, writer_dir=None, suffix=None):
        if writer_dir:
            self.writer = SummaryWriter(writer_dir, filename_suffix=suffix)
        else:
            self.writer = SummaryWriter()

    # save train loss, f1, iou, current lr
    def add_train_scaler(self, i, epoch, epoch_len, loss, f1, iou, lr):
        total_step = i + epoch * epoch_len
        self.writer.add_scalar('train/loss', loss, total_step)
        self.writer.add_scalar('train/f1', f1, total_step)
        self.writer.add_scalar('train/iou', iou, total_step)
        self.writer.add_scalar('train/lr', lr, total_step)

    # save val loss, f1, iou
    def add_val_scaler(self, i, epoch, epoch_len, loss, f1, iou):
        total_step = i + epoch * epoch_len
        self.writer.add_scalar('val/loss', loss, total_step)
        self.writer.add_scalar('val/f1', f1, total_step)
        self.writer.add_scalar('val/iou', iou, total_step)

    def close(self):
        self.writer.close()