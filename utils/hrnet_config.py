# ----------------------------------------------------------- #
#  This is code confidential, for peer-review purposes only   #
#  and protected under conference code of ethics              #
# ----------------------------------------------------------- #

# Code adapted from https://github.com/HRNet/HRNet-Human-Pose-Estimation authored by Bin Xiao

class empty_class:
    pass

class cfg:
    MODEL = empty_class()
    MODEL.NAME = 'pose_hrnet'
    MODEL.INIT_WEIGHTS = True
    MODEL.PRETRAINED = ''
    MODEL.NUM_JOINTS = 17
    MODEL.TAG_PER_JOINT = True
    MODEL.TARGET_TYPE = 'gaussian'
    MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
    MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
    MODEL.SIGMA = 2
    MODEL.EXTRA = empty_class()
    MODEL.EXTRA.PRETRAINED_LAYERS = ['*']
    MODEL.EXTRA.STEM_INPLANES = 64
    MODEL.EXTRA.FINAL_CONV_KERNEL = 1

    MODEL.EXTRA.STAGE2 = empty_class()
    MODEL.EXTRA.STAGE2.NUM_MODULES = 1
    MODEL.EXTRA.STAGE2.NUM_BRANCHES = 2
    MODEL.EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
    MODEL.EXTRA.STAGE2.NUM_CHANNELS = [32, 64]
    MODEL.EXTRA.STAGE2.BLOCK = 'BASIC'
    MODEL.EXTRA.STAGE2.FUSE_METHOD = 'SUM'

    MODEL.EXTRA.STAGE3 = empty_class()
    MODEL.EXTRA.STAGE3.NUM_MODULES = 1
    MODEL.EXTRA.STAGE3.NUM_BRANCHES = 3
    MODEL.EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
    MODEL.EXTRA.STAGE3.NUM_CHANNELS = [32, 64, 128]
    MODEL.EXTRA.STAGE3.BLOCK = 'BASIC'
    MODEL.EXTRA.STAGE3.FUSE_METHOD = 'SUM'

    MODEL.EXTRA.STAGE4 = empty_class()
    MODEL.EXTRA.STAGE4.NUM_MODULES = 1
    MODEL.EXTRA.STAGE4.NUM_BRANCHES = 4
    MODEL.EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    MODEL.EXTRA.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    MODEL.EXTRA.STAGE4.BLOCK = 'BASIC'
    MODEL.EXTRA.STAGE4.FUSE_METHOD = 'SUM'



