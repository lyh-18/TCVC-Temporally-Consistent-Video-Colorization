import torch


import models.archs.TCVC_IDC_arch as TCVC_IDC_arch



# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model_G"]

    # image restoration
    if which_model == "TCVC_IDC":
        print("which model: {}".format(which_model))
        netG = TCVC_IDC_arch.TCVC_IDC(
            nf=opt_net["nf"],
            N_RBs=opt_net["N_RBs"],
            key_net=opt_net["key_net"],
            dataset=opt_net["DAVIS"],
            train_flow_keyNet=opt_net["train_flow_keyNet"],
        )
    else:
        raise NotImplementedError(
            "Generator model [{:s}] not recognized".format(which_model)
        )

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt["network_D"]
    which_model = opt_net["which_model_D"]

    if which_model == "discriminator_vgg_128":
        netD = SRGAN_arch.Discriminator_VGG_128(
            in_nc=opt_net["in_nc"], nf=opt_net["nf"]
        )
    else:
        raise NotImplementedError(
            "Discriminator model [{:s}] not recognized".format(which_model)
        )
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt["gpu_ids"]
    device = torch.device("cuda" if gpu_ids else "cpu")
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )
    netF.eval()  # No need to train
    return netF
