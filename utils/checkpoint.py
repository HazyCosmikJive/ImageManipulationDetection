# checkpoint related
# https://github.com/amazon-research/video-contrastive-learning/blob/main/utils/util.py
import re
import os
import torch
import glob


# TODO: auto resume from last checkpoint
# TODO: when auto resume, resume from checkpoint's epoch
# TODO: process unmatched keys
def load_checkpointpath(
    config,
    logger,
    model,
    optimizer=None,
    scheduler=None,
    testmode=False,
    resume_last=False,
    resume_best=False,
    path=None,
):
    if resume_best and path is None:
        path = os.path.join(config.common.checkpointpath, "best.pth.tar")
    if resume_last and path is None:
        pathlist = glob.glob(config.common.checkpointpath + "/epoch*.pth.tar")
        # sort with epoch num, choose latest epoch to resume
        pathlist = sorted(pathlist, key=lambda name: int(name.split("/")[-1].split("_")[0][5:]))
        path = pathlist[-1]
    elif path is None:  # load from a given path when path is not None
        logger.info("No checkpoint path provided. Allow resume_best / resume_last or provided a checkpoint path.(｡O_O｡)")
        return model
    assert os.path.exists(path), "Checkpoint {} do not exist".format(path)


    checkpoint = torch.load(path, map_location='cpu')
    logger.info("[LOAD MODEL] load from {}, epoch {}".format(path, checkpoint["epoch"]))
    # model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint["model_dict"].items()})
    model.load_state_dict(checkpoint["model_dict"])
    logger.info("- model dict loaded")
    
    if not testmode:
        if optimizer is not None:
            pass # TODO
        if scheduler is not None:
            pass
    del checkpoint
    # torch.cuda.empty_cache()


def load_pretrained():
    pass


def save_checkpoint(model_savepath, epoch, model, optimizer, scheduer, f1, best_f1, best_epoch, is_best=False):
    if is_best:
        torch.save({
                "epoch": epoch,
                "model_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduer.state_dict(),
                "f1": f1,
                "best_f1": best_f1,
                "best_epoch": best_epoch
            }, os.path.join(model_savepath, "best.pth.tar"))
    else:
        torch.save({
                "epoch": epoch,
                "model_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduer.state_dict(),
                "f1": f1,
                "best_f1": best_f1,
                "best_epoch": best_epoch
        }, os.path.join(model_savepath, "epoch%d_f1_%.4f.pth.tar" % (epoch, f1)))

