import os
import numpy as np
import cv2
import glob

IMAGE_NAME = "2008_003020"

VOC_PATH = "/home/kevin/data/VOC2012"

image_path = os.path.join(VOC_PATH, "JPEGImages", "{}.jpg".format(IMAGE_NAME))
print(image_path)

bbox_info_folder = os.path.join(VOC_PATH, "extracted", "{}".format(IMAGE_NAME))

info_files = glob.glob(os.path.join(bbox_info_folder, "*.txt"))
print(info_files)

image = cv2.imread(image_path)

for info_file in info_files:
    f = open(info_file)
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        print(tokens)
        cv2.rectangle(image,
                      (int(float(tokens[0])), int(float(tokens[1]))),
                      (int(float(tokens[2])), int(float(tokens[3]))),
                      (255, 255, 0), 1)

    f.close()

cv2.imshow("t", image)
cv2.waitKey(0)