import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = (300, 600)
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# 300x300
# fm_size, downscale_ratio, bbox sizes, [aspect ratios]
specs_300_300 = [
    SSDSpec((19, 19), 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec((10, 10), 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec((5, 5), 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec((3, 3), 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec((2, 2), 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec((1, 1), 300, SSDBoxSizes(285, 330), [2, 3])
]

'''
torch.Size([1, 3, 600, 300])
torch.Size([1, 32, 300, 150])
torch.Size([1, 16, 300, 150])
torch.Size([1, 24, 150, 75])
torch.Size([1, 24, 150, 75])
torch.Size([1, 32, 75, 38])
torch.Size([1, 32, 75, 38])
torch.Size([1, 32, 75, 38])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 19, 10])
torch.Size([1, 576, 19, 10])
torch.Size([1, 576, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 320, 19, 10])
torch.Size([1, 1280, 19, 10])
torch.Size([1, 512, 10, 5])
torch.Size([1, 256, 5, 3])
torch.Size([1, 256, 3, 2])
torch.Size([1, 64, 2, 1])
'''

# 300x600
specs_300_600 = [
    SSDSpec((38, 19), 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec((19, 10), 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec((10, 5), 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec((5, 3), 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec((3, 2), 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec((2, 1), 300, SSDBoxSizes(285, 330), [2, 3])
]

'''
torch.Size([1, 3, 600, 300])
torch.Size([1, 32, 300, 150])
torch.Size([1, 16, 300, 150])
torch.Size([1, 24, 150, 75])
torch.Size([1, 24, 150, 75])
torch.Size([1, 32, 75, 38])
torch.Size([1, 32, 75, 38])
torch.Size([1, 32, 75, 38])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 64, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 96, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 38, 19])
torch.Size([1, 576, 19, 10])
torch.Size([1, 576, 19, 10])
torch.Size([1, 576, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 160, 19, 10])
torch.Size([1, 320, 19, 10])
torch.Size([1, 1280, 19, 10])
torch.Size([1, 512, 10, 5])
torch.Size([1, 256, 5, 3])
torch.Size([1, 256, 3, 2])
torch.Size([1, 64, 2, 1])
'''


priors = generate_ssd_priors(specs_300_600, image_size)