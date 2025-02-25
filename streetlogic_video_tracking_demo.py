import vision.ssd.ssd
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.config import mobilenetv1_ssd_config as config

from tracking.tracker import Tracker

import cv2
import sys
import numpy as np


Y_OFFSET_START = 370
X_OFFSET_START = 100


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <model path> <label path> <video path> <output path>')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
video_path = sys.argv[3]

output_path = video_path[:-4] + "_output.mp4"

class_names = [name.strip() for name in open(label_path).readlines()]


net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

tracker = Tracker(num_frames_keep=5)

colors_list = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 128),
    (128, 0, 255),
    (128, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 10, (1920, 1080))

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    display_image = frame.copy()

    input_image = frame[Y_OFFSET_START:Y_OFFSET_START+config.image_size[1], X_OFFSET_START:X_OFFSET_START+config.image_size[0]]
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(display_image,
                  (X_OFFSET_START, Y_OFFSET_START),
                  (X_OFFSET_START+config.image_size[0], Y_OFFSET_START+config.image_size[1]),
                  (0, 0, 0), 4)

    boxes, labels, probs, features = predictor.predict(input_image, 10, 0.4, output_image_features=True)

    # Remove bad detections and classes we don't care about.
    boxes_filtered = []
    labels_filtered = []
    features_filtered = []
    probs_filtered = []
    for i in range(boxes.size(0)):
        if class_names[labels[i]] != 'car' or probs[i] < 0.7:
            continue
        boxes_filtered.append(boxes[i])
        labels_filtered.append(labels[i])
        features_filtered.append(features[i])
        probs_filtered.append(probs[i])

    track_ids = tracker.process(boxes_filtered, features_filtered)


    for i in range(len(boxes_filtered)):
        box = [int(x) for x in boxes_filtered[i]]
        box_offset = [
            box[0] + X_OFFSET_START,
            box[1] + Y_OFFSET_START,
            box[2] + X_OFFSET_START,
            box[3] + Y_OFFSET_START,
        ]

        color = colors_list[track_ids[i] % len(colors_list)]

        cv2.rectangle(display_image, (box_offset[0], box_offset[1]), (box_offset[2], box_offset[3]), color, 4)
        label = "ID-{} F{}".format(track_ids[i], features_filtered[i].shape[0])
        cv2.putText(display_image, label,
                    (box_offset[0] + 10, int((box_offset[1] + box_offset[3])/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,  # font scale
                    color,
                    1)  # line type


    out.write(display_image)

    # cv2.imshow('frame', display_image)
    # if cv2.waitKey(0) == ord('q'):
    #     break

out.release()