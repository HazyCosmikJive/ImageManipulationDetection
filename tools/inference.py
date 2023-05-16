import torch
import torchvision.transforms as transforms

import cv2
import os
import numpy as np
from torch_utils import AverageMeter

from data.dataset_entry import get_test_loader
from utils.metric import calculate_f1iou
from utils.checkpoint import load_checkpointpath
from utils.dist import get_rank, get_world_size, all_reduce_numpy

import ipdb

transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

def inference(config, logger, model, device):
    load_checkpointpath(config, logger, model, testmode=True, resume_best=True)

    test_loader = get_test_loader(config, logger)
    
    test_f1s = AverageMeter(name="TestF1", length=config.trainer.print_freq)
    test_ious = AverageMeter(name="TestIoU", length=config.trainer.print_freq)
    test_f1_list = []
    test_iou_list = []

    model.eval()
    imgtypelist = []
    imgpathlist = []
    f1_fulllist = []
    iou_fulllist = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch["image"]
            masks = batch["mask"]
            images, masks = images.to(device), masks.to(device)
            imgtypes = batch["imgtype"]
            imgpaths = batch["imgpath"]
            imgtypelist.extend(imgtypes)
            imgpathlist.extend(imgpaths)

            out = model(images)["masks"]
            out = torch.sigmoid(out)

            # save pred and pred_th
            imgs = [np.array(transform_pil(model_out)) for model_out in out]
            imgs_th = [255.0 * (img > 255 * config.infer.get("th", 0.5)) for img in imgs]

            pred_dir = os.path.join(config.common.predpath, "ori")
            pred_th_dir = os.path.join(config.common.predpath, "th{}".format(str(config.infer.get("th", 0.5))))
            if get_rank() == 0:
                os.makedirs(pred_dir, exist_ok=True)
                os.makedirs(pred_th_dir, exist_ok=True)
            for idx in range(len(imgs)):
                imgname = imgpaths[idx].split("/")[-1].split(".")[0] + ".png"
                oriimg = cv2.imread(imgpaths[idx])
                cv2.imwrite(os.path.join(pred_dir, imgname), cv2.resize(imgs[idx], (oriimg.shape[1], oriimg.shape[0])))
                cv2.imwrite(os.path.join(pred_th_dir, imgname), cv2.resize(imgs_th[idx], (oriimg.shape[1], oriimg.shape[0])))

            f1, iou = calculate_f1iou(out.cpu(), masks.cpu(), th=config.infer.get("th", 0.5))

            f1_fulllist.extend(f1)
            iou_fulllist.extend(iou)

            f1 = np.mean(f1)
            iou = np.mean(iou)
            test_f1s.update(f1)
            test_ious.update(iou)
            test_f1_list.append(f1)
            test_iou_list.append(iou)

            if i % config.trainer.print_freq == 0:
                logger.info("TEST Iter [{}/{}] | F1 {:.4f} IoU {:.4f}".format(i, len(test_loader), test_f1s.avg, test_ious.avg))

    if config.common.get("dist", False):
        # save f1, iou, imgtype from all ranks to txt files
        filepath = os.path.join(config.common.predpath, "th{}.rank_{}.txt".format(str(config.infer.get("th", 0.5)), str(get_rank())))
        f = open(filepath, "w")
        for i in range(len(imgpathlist)):
            # imgname, f1, iou, imgtype (at th)
            line = imgpathlist[i].split("/")[-1].split(".")[0] + " " + "{:.4f}".format(f1_fulllist[i]) + " {:.4f}".format(iou_fulllist[i]) + " " + imgtypelist[i]
            f.write(line + "\n")
        f.close()

        # gather f1 and iou result
        f1_np = np.array(test_f1_list)
        iou_np = np.array(test_iou_list)

        f1_reduce = all_reduce_numpy(f1_np) / get_world_size()
        iou_reduce = all_reduce_numpy(iou_np) / get_world_size

        return f1_reduce.mean(), iou_reduce.mean()

    else:
        # save f1, iou, imgtype to a txt file
        filepath = os.path.join(config.common.predpath, "th{}.txt".format(str(config.infer.get("th", 0.5))))
        f = open(filepath, "w")
        for i in range(len(imgpathlist)):
            # imgname, f1, iou, imgtype (at th)
            line = imgpathlist[i].split("/")[-1].split(".")[0] + " " + "{:.4f}".format(f1_fulllist[i]) + " {:.4f}".format(iou_fulllist[i]) + " " + imgtypelist[i]
            f.write(line + "\n")
        f.close()

        return sum(test_f1_list) / len(test_f1_list), sum(test_iou_list) / len(test_iou_list)