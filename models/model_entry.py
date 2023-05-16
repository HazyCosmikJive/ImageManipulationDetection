import segmentation_models_pytorch as smp

from models.hrnet import HighResolutionNet
from models.smp_models.deeplabv3 import DeepLabV3Plus
from models.smp_models.deeplab_srm import DeepLabV3Plus as DeepLabV3Plus_srm

def model_entry(config, logger):
    model = None
    if "unetplusplus" == config.model.architecture.lower():
        model = smp.UnetPlusPlus(
            encoder_name=config.model.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            decoder_attention_type="scse",
            classes=1,
        )
    elif "deeplab" == config.model.architecture.lower():  # simple deeplabv3+
        model = DeepLabV3Plus(
            config=config,
            encoder_name=config.model.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    elif "deeplab_srm" == config.model.architecture.lower():  # with srm noise stream
        model = DeepLabV3Plus_srm(
            encoder_name=config.model.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    elif "hrnet" in config.model.architecture.lower():
        model = HighResolutionNet(
            config=config,
            num_classes=1,
        )
    else:
        raise Exception("Failed to build a model (-_-)")
    logger.info("Use Model {} with encoder {}".format(config.model.architecture, config.model.encoder))
    logger.info("[MODEL] Build Model Done.")

    return model
