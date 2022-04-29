import cv2
import numpy as np
import os
import glob
import pandas as pd
import torch

VOC_PATH = "/home/kevin/data/VOC2012/"
EXTRACTED_DATA_PATH = os.path.join(VOC_PATH, "extracted")

image_names = os.listdir(EXTRACTED_DATA_PATH)

image_str_list = []
nms_cluster_id_list = []
class_id_list = []
score_list = []
box_id_list = []
fm_id_list = []
box_fm_index = []
feature_length = []

for image_index, image_name in enumerate(image_names):
    cluster_files = glob.glob(os.path.join(EXTRACTED_DATA_PATH, image_name, "*.txt"))

    for cluster_file in cluster_files:
        cluster_id = int(os.path.basename(cluster_file)[:-4])

        box_feature_files = glob.glob(os.path.join(EXTRACTED_DATA_PATH, image_name, "{}_*.pt".format(cluster_id)))
        box_feature_ids = [os.path.basename(x).split("_")[1][:-3] for x in box_feature_files]
        # print(box_feature_files)
        # print(box_feature_ids)
        id_to_file_dict = {}
        for id, file in zip(box_feature_ids, box_feature_files):
            id_to_file_dict[int(id)] = file

        # print(id_to_file_dict)

        f = open(cluster_file, "r")
        lines = f.readlines()
        f.close()
        for box_id, line in enumerate(lines):
            tokens = line.split()
            # print(tokens)
            image_str_list.append(image_name)
            nms_cluster_id_list.append(cluster_id)
            class_id_list.append(int(tokens[4]))
            score_list.append(float(tokens[5]))
            box_id_list.append(box_id)
            fm_id_list.append(int(tokens[6]))
            box_fm_index.append(int(tokens[7]))

            fm_file = id_to_file_dict[box_id]
            # print(fm_file)
            fm_tensor = torch.load(fm_file)
            feature_length.append(list(fm_tensor.shape)[0])

    if image_index > 0 and image_index % 1000 == 0:
        print("finish {} of {}".format(image_index, len(image_names)))

    # break

df = pd.DataFrame({
    'image': image_str_list,
    'cluster_id': nms_cluster_id_list,
    'class_id': class_id_list,
    'score': score_list,
    'box_id': box_id_list,
    'fm_id': fm_id_list,
    'box_fm_index': box_fm_index,
    'fv_length': feature_length
})

print(df)

df.to_csv(os.path.join(VOC_PATH, "box_features.csv"))