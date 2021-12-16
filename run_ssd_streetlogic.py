import vision.ssd.ssd
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.config import mobilenetv1_ssd_config as config

import cv2
import sys


Y_OFFSET_START = 370
X_OFFSET_START = 700


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <model path> <label path> <image path>')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
image_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]


net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
crop_image = orig_image[Y_OFFSET_START:Y_OFFSET_START+config.image_size[1], X_OFFSET_START:X_OFFSET_START+config.image_size[0]]
image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

cv2.rectangle(orig_image,
              (X_OFFSET_START, Y_OFFSET_START),
              (X_OFFSET_START+config.image_size[0], Y_OFFSET_START+config.image_size[1]),
              (0, 255, 0), 4)

for i in range(boxes.size(0)):
    box = [int(x) for x in boxes[i, :]]
    box_offset = [
        box[0] + X_OFFSET_START,
        box[1] + Y_OFFSET_START,
        box[2] + X_OFFSET_START,
        box[3] + Y_OFFSET_START,
    ]
    cv2.rectangle(orig_image, (box_offset[0], box_offset[1]), (box_offset[2], box_offset[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box_offset[0] + 20, box_offset[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
