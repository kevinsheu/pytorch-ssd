import torch
import cv2
import os
import numpy as np

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.box_utils import get_iou_numpy


CLASS_NAMES_FILE = "models/voc-model-labels.txt"
MODEL_FILE = "models/mb2-ssd-lite-mp-0_686.pth"
VOC_PATH = "/home/kevin/data/VOC2012"
DETECTION_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5


def output_index_to_feature_map_id(box_index, image_features):
    running_sum = 0
    feature_map_index = 0
    for image_feature_tensor in image_features:
        if box_index < running_sum + image_features[feature_map_index].shape[0]:
            if feature_map_index == 0:
                return 0, box_index
            return feature_map_index, box_index % running_sum
        running_sum += image_features[feature_map_index].shape[0]
        feature_map_index += 1

    assert False

def get_nms_assignments(box_vectors):

    visited = np.zeros(box_vectors.shape[0])
    assignments = np.full(box_vectors.shape[0], -1)

    for ii in range(visited.shape[0]):
        if visited[ii] == 1:
            continue
        assignments[ii] = ii
        for jj in range(ii+1, visited.shape[0]):
            iou = get_iou_numpy(box_vectors[ii, :4], box_vectors[jj, :4])
            if iou < IOU_THRESHOLD:
                continue
            assignments[jj] = ii
            visited[jj] = 1

        visited[ii] = 1

    return assignments


class_names = [name.strip() for name in open(CLASS_NAMES_FILE).readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(MODEL_FILE)
predictor = create_mobilenetv2_ssd_lite_predictor(net,
                                                  candidate_size=200,
                                                  device=torch.device('cuda:0'),
                                                  resize=True)

image_dir = os.path.join(VOC_PATH, "JPEGImages")
extraction_path = os.path.join(VOC_PATH, "extracted")
if not os.path.exists(extraction_path):
    os.mkdir(extraction_path)

num_images = len(os.listdir(image_dir))

for image_index, image_name_full in enumerate(os.listdir(image_dir)):
    # if image_index < 2:
    #     continue
    image_path = os.path.join(image_dir, image_name_full)
    image_original = cv2.imread(image_path)
    image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    scores, boxes, image_features = predictor.predict(image, 10, 0.4, output_image_features=True, return_all=True)
    boxes[:, 0] *= image.shape[1]
    boxes[:, 1] *= image.shape[0]
    boxes[:, 2] *= image.shape[1]
    boxes[:, 3] *= image.shape[0]

    image_name = image_name_full[:-4]
    extracted_image_path = os.path.join(extraction_path, image_name)
    if not os.path.exists(extracted_image_path):
        os.mkdir(extracted_image_path)

    output_box_vectors = []
    # Test visualization
    for i in range(boxes.shape[0]):
        class_index = torch.argmax(scores[i])
        if class_index == 0:
            continue
        if scores[i][class_index] < DETECTION_THRESHOLD:
            continue

        fm_index, box_i = output_index_to_feature_map_id(i, image_features)

        # cv2.rectangle(image_original,
        #               (int(boxes[i][0]), int(boxes[i][1])),
        #               (int(boxes[i][2]), int(boxes[i][3])),
        #               (255, 255, 0), 1)

        data_vector = boxes[i].numpy()
        data_vector = np.append(data_vector, class_index.numpy())
        data_vector = np.append(data_vector, scores[i][class_index].numpy())
        data_vector = np.append(data_vector, fm_index)
        data_vector = np.append(data_vector, box_i)

        output_box_vectors.append(data_vector)

    output_box_vectors = np.array(output_box_vectors)
    if output_box_vectors.shape[0] == 0:
        continue
    output_box_vectors = output_box_vectors[output_box_vectors[:, 5].argsort()[::-1]]

    nms_assignments = get_nms_assignments(output_box_vectors)
    groups = np.unique(nms_assignments)

    counter = {}
    for group_id in groups:
        counter[group_id] = 0
        group_text_file_path = os.path.join(extracted_image_path, "{}.txt".format(group_id))
        if os.path.exists(group_text_file_path):
            os.remove(group_text_file_path)

    for i in range(output_box_vectors.shape[0]):
        box_group = nms_assignments[i]
        group_text_file_path = os.path.join(extracted_image_path, "{}.txt".format(box_group))
        file = open(group_text_file_path, "a")
        file.write("{} {} {} {} {} {} {} {}\n".format(output_box_vectors[i][0],
                                                      output_box_vectors[i][1],
                                                      output_box_vectors[i][2],
                                                      output_box_vectors[i][3],
                                                      int(output_box_vectors[i][4]),
                                                      output_box_vectors[i][5],
                                                      int(output_box_vectors[i][6]),
                                                      int(output_box_vectors[i][7])))
        file.close()

        feature_vector_file = os.path.join(extracted_image_path, "{}_{}.npy".format(box_group, counter[box_group]))
        feature_vector = image_features[int(output_box_vectors[i][6])][int(output_box_vectors[i][7])].numpy()
        np.save(feature_vector_file, feature_vector)

        counter[box_group] += 1

    # cv2.imshow("t", image_original)
    # cv2.waitKey(0)
    # cv2.imwrite("/home/kevin/example.png", image_original)

    if image_index > 0 and image_index % 100 == 0:
        print("finish {} of {}".format(image_index, num_images))

    # if image_index > 1:
    #     break