import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config):
    return A.Compose([
        A.Resize(config.data.image_size, config.data.image_size, p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.2, rotate_limit=20, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(transpose_mask=True),
    ], p=1.)


# only normalize in validation
def get_val_transforms(config):
    return A.Compose([
        A.Resize(config.data.image_size, config.data.image_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(transpose_mask=True),
    ], p=1.)