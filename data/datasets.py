import cv2
import numpy as np
from torch.utils.data import Dataset


class PSDataset(Dataset):
    def __init__(
            self,
            root_dir,
            infolist,
            transforms=None,
            mode="train",
        ):
        self.root_dir = root_dir
        self.imgtypelist = []
        self.transforms = transforms
        self.mode = mode

        self._parse_info(infolist)

    def _parse_info(self, infolist):
        self.imglist = [self.root_dir + each.split(" ")[0] for each in infolist]
        self.masklist = [self.root_dir + each.split(" ")[1] if "None" not in each.split(" ")[1] else each.split(" ")[1] for each in infolist]
        self.imgtypelist = [each.split(" ")[2] for each in infolist]

    def __getitem__(self, index):
        # load img
        imgpath = self.imglist[index]
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        maskpath = self.masklist[index]
        if "None" in maskpath:  # for some authentic images, mask is None
            mask = np.zeros((img.shape[0], img.shape[1]))
        else:
            mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE) / 255.         

        if self.mode == "train":
            # transform
            augments = self.transforms(image=img, mask=mask)
            img = augments["image"].float()
            mask = augments["mask"][None, ...].float()

            return {
                "image": img,
                "mask": mask,
                "imgtype": self.imgtypelist[index],
            }
        
        elif self.mode == "test":
            augments = self.transforms(image=img, mask=mask)
            img = augments["image"].float()
            mask = augments["mask"][None, ...].float()
            
            return {
                "image": img,
                "mask": mask,
                "imgtype": self.imgtypelist[index],
                "imgpath": imgpath,
            }
        else:
            raise NotImplementedError("Wrong mode, `train` and `test` only.")

    def __len__(self):
        return len(self.imglist)