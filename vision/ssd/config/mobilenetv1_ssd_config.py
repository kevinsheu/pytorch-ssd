import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = (300, 300)
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

# 600x300
specs_600_300 = [
    SSDSpec((19, 38), 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec((10, 19), 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec((5, 10), 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec((3, 5), 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec((2, 3), 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec((1, 2), 300, SSDBoxSizes(285, 330), [2, 3])
]

specs_600_600 = [
    SSDSpec((38, 38), 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec((19, 19), 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec((10, 10), 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec((5, 5), 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec((3, 3), 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec((2, 2), 300, SSDBoxSizes(285, 330), [2, 3])
]


priors = generate_ssd_priors(specs_300_300, image_size)