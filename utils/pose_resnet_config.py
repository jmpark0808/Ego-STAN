class empty_class:
    pass

class cfg:

    # pose_resnet related params
    POSE_RESNET = empty_class()
    POSE_RESNET.NUM_LAYERS = 50
    POSE_RESNET.DECONV_WITH_BIAS = False
    POSE_RESNET.NUM_DECONV_LAYERS = 3
    POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
    POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
    POSE_RESNET.FINAL_CONV_KERNEL = 1
    POSE_RESNET.TARGET_TYPE = 'gaussian'
    POSE_RESNET.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    POSE_RESNET.SIGMA = 2

    MODEL_EXTRAS = {
        'pose_resnet': POSE_RESNET,
    }

    # common params for NETWORK
    MODEL = empty_class()
    MODEL.NAME = 'pose_resnet'
    MODEL.INIT_WEIGHTS = True
    MODEL.PRETRAINED = ''
    MODEL.NUM_JOINTS = 17
    MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    MODEL.EXTRA = MODEL_EXTRAS[MODEL.NAME]

    MODEL.STYLE = 'pytorch'



