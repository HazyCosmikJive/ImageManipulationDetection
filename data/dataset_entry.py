import torch
from torch.utils.data import DataLoader

from .datasets import PSDataset
from .transforms import get_train_transforms, get_val_transforms
from utils.dist import get_rank, get_world_size


def process_datainfo(config):
    with open(config.data.train_metafile, "r") as f:
        train_infolist = f.readlines()
    with open(config.data.val_metafile, "r") as f:
        val_infolist = f.readlines()
    train_infolist = [each.strip("\n") for each in train_infolist]
    val_infolist = [each.strip("\n") for each in val_infolist]

    return train_infolist, val_infolist

def get_train_val_loader(config, logger):
    train_infolist, val_infolist = process_datainfo(config)
    logger.info("[DATA INFO] Trainset: {} | Valset: {}".format(len(train_infolist), len(val_infolist)))

    train_set = PSDataset(
        root_dir=config.data.train_root,
        infolist=train_infolist,
        transforms=get_train_transforms(config),
        mode="train",
    )
    val_set = PSDataset(
        root_dir=config.data.train_root,
        infolist=val_infolist,
        transforms=get_val_transforms(config),
        mode="train",
    )

    train_sampler = None
    val_sampler = None
    if config.common.get("dist", False):
        train_sampler = torch.utils.data.DistributedSampler(train_set)
        val_sampler = torch.utils.data.DistributedSampler(val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=config.trainer.batchsize,
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=True if train_sampler is None else False,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.trainer.batchsize,
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=False,
        sampler=val_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )      
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
    }

def get_test_loader(config, logger):
    with open(config.data.test_metafile, "r") as f:
        test_infolist = f.readlines()
    test_infolist = [each.strip("\n") for each in test_infolist]

    logger.info("[DATA INFO] Testset: {}".format(len(test_infolist)))

    test_set = PSDataset(
        root_dir=config.data.test_root,
        infolist=test_infolist,
        transforms=get_val_transforms(config),
        mode="test",
    )

    test_sampler = None
    if config.common.get("dist", False):
        test_sampler = torch.utils.data.DistributedSampler(test_set)


    test_loader = DataLoader(
        test_set,
        batch_size=config.infer.get("batchsize", 8),
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=False,
        sampler=test_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )

    return test_loader