import os
import torch
from torch.cuda.amp import GradScaler

import mmcv
import numpy as np
import datetime
from torch_utils import AverageMeter

from utils.metric import calculate_f1iou
from utils.checkpoint import load_checkpointpath, save_checkpoint
from utils.dist import all_reduce_numpy, get_rank, get_world_size


def train_model(
    model,
    config,
    device,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    logger,
    writer=None,
    train_sampler=None,
    val_sampler=None,
):

    best_f1 = 0
    best_epoch = 0
    scaler = GradScaler()

    for epoch in range(1, config.trainer.epoch + 1):
        if train_sampler is not None: # dist
            train_sampler.set_epoch(epoch)
        timer = mmcv.Timer()
        losses = AverageMeter(name="TrainLoss", length=config.trainer.print_freq)
        f1s = AverageMeter(name="TrainF1s", length=config.trainer.print_freq)
        ious = AverageMeter(name="TrainIoUs", length=config.trainer.print_freq)
        batch_time = AverageMeter(name="BatchTime", length=config.trainer.print_freq)
        
        model.train()
        f1_list = []
        iou_list = []
        loss_list = []
        
        for i, batch in enumerate(train_loader):
            images = batch["image"]
            masks = batch["mask"]
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                out = model(images)
                predmasks = out["masks"]
                predmasks = torch.sigmoid(predmasks)
                if config.model.get("use_contrast", False):
                    with_feature = (epoch >= config.model.contrast.get("warmup_epoch", 5))
                    loss = criterion(out, masks, with_feature)
                else:
                    loss = criterion(predmasks, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.update(loss.item())

            f1, iou = calculate_f1iou(predmasks.cpu().detach(), masks.cpu().detach(), th=config.data.th)

            if config.common.get("dist", False):
                # dist: reduce result from multi gpus
                # print(">> before reduce, rank {}\nf1\n {}".format(get_rank(), f1))
                f1_reduce = all_reduce_numpy(f1) / get_world_size()
                iou_reduce = all_reduce_numpy(iou) / get_world_size()
                # print("after reduce, rank: {} \nf1\n {}".format(get_rank(), f1_reduce))

                f1 = np.mean(f1_reduce)
                iou = np.mean(iou_reduce)

            f1 = np.mean(f1)
            iou = np.mean(iou)
            f1s.update(f1)
            ious.update(iou)
            batch_time.update(timer.since_last_check())
            loss_list.append(losses.avg)
            f1_list.append(f1s.avg)
            iou_list.append(ious.avg)

            if i % config.trainer.print_freq == 0:
                eta_seconds = int(batch_time.avg * (len(train_loader) * (config.trainer.epoch - epoch + 1) - i))
                eta = str(datetime.timedelta(seconds=eta_seconds))
                logger.info("Epoch {} | TRAIN Iter [{}/{}] | Loss {:.4f} F1 {:.4f} IoU {:.4f} | lr: {:.6f} batchtime: {:.4f} ETA: {}".format(
                    epoch, i, len(train_loader), losses.avg, f1s.avg, ious.avg, optimizer.param_groups[0]['lr'], batch_time.avg, eta
                ))
                if writer is not None:
                    writer.add_train_scaler(i, epoch, len(train_loader), losses.avg, f1s.avg, ious.avg, optimizer.param_groups[0]['lr'])

        # avg stats for a epoch
        train_loss = sum(loss_list) / len(loss_list)
        train_f1 = sum(f1_list) / len(f1_list)
        train_iou = sum(iou_list) / len(iou_list)

        # eval
        val_f1, val_iou = eval_model(model, config, epoch, device, criterion, val_loader, logger, writer, val_sampler)

        # save model
        if config.common.get("debug", False):  # debug mode
            model_savepath = config.common.checkpointpath_debug
        else:
            model_savepath = config.common.checkpointpath

        if get_rank() == 0:
            if epoch % config.trainer.save_freq == 0:
                save_checkpoint(model_savepath, epoch, model, optimizer, scheduler, val_f1, best_f1, best_epoch)
            # update best model
            if val_f1 >= best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                save_checkpoint(model_savepath, epoch, model, optimizer, scheduler, val_f1, best_f1, best_epoch, is_best=True)
        
        logger.info("\n===> Epoch {} \n     Train Loss {:.4f} F1 {:.4f} IoU {:.4f} \n     Val F1 {:.4f} Val IoU: {:.4f} \n     * Best F1: {:.4f} at epoch {:.4f}".format(
                            epoch,
                            train_loss, train_f1, train_iou,
                            val_f1, val_iou,
                            best_f1, best_epoch,
                        ))
            

def eval_model(
    model,
    config,
    epoch,
    device,
    criterion,
    val_loader,
    logger,
    writer=None,
    val_sampler=None,
):
    val_losses = AverageMeter(name="ValLoss", length=config.trainer.print_freq)
    val_f1s = AverageMeter(name="ValF1", length=config.trainer.print_freq)
    val_ious = AverageMeter(name="ValIou", length=config.trainer.print_freq)
    val_f1_list = []
    val_iou_list = []
    model.eval()

    if val_sampler is not None:
        val_sampler.set_epoch(epoch)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"]
            masks = batch["mask"]
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            predmasks = out["masks"]
            predmasks = torch.sigmoid(predmasks)

            if config.model.get("use_contrast", False):
                loss = criterion(out, masks, with_feature=False) # only compute BCELoss for validation
            else:
                loss = criterion(predmasks, masks)
            val_losses.update(loss.item())

            f1, iou = calculate_f1iou(predmasks.cpu(), masks.cpu(), th=config.data.th)

            if config.common.get("dist", False):
                # dist: reduce result from multi gpus
                f1_reduce = all_reduce_numpy(f1) / get_world_size()
                iou_reduce = all_reduce_numpy(iou) / get_world_size()

                f1 = np.mean(f1_reduce)
                iou = np.mean(iou_reduce)
            f1 = np.mean(f1)
            iou = np.mean(iou)
            val_f1s.update(f1)
            val_f1_list.append(f1)
            val_ious.update(iou)
            val_iou_list.append(iou)

            if i % config.trainer.print_freq == 0:
                logger.info("Epoch {} | EVAL Iter [{}/{}] | Loss {:.4f} F1 {:.4f} IoU {:.4f} ".format(epoch, i, len(val_loader), val_losses.avg, val_f1s.avg, val_ious.avg))
                if writer is not None:
                    writer.add_val_scaler(i, epoch, len(val_loader), val_losses.avg, val_f1s.avg, val_ious.avg)

    return sum(val_f1_list) / len(val_f1_list), sum(val_iou_list) / len(val_iou_list)