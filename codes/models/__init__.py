import logging

logger = logging.getLogger("base")


def create_model(opt):
    model = opt["model"]
    # image restoration
    if model == "video_colorization_warp2":
        from .video_colorization_model_warploss2 import VideoColorizationModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
