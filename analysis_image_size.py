import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

VOC_PATH = "/home/kevin_streetlogic_ai/data/VOC2012"
OUTPUT_PATH = "/home/kevin_streetlogic_ai/code/output/"


def ParseImageSetFile(set_file_path):
    image_names = []
    f = open(set_file_path, 'r')
    lines = f.read().splitlines()
    for line in lines:
        tokens = line.split()
        img_path = tokens[0]
        status = int(tokens[1])
        if status == -1:
            continue

        image_names.append(img_path)

    return image_names


def GetImageDimensionsHeatmap(image_names, fig_name):
    print("Checking {} images...".format(len(image_names)))

    image_sizes = []
    for i, image_name in enumerate(image_names):
        img = cv2.imread(os.path.join(VOC_PATH, "JPEGImages", image_name))
        image_sizes.append(img.shape[:2])
        if i > 0 and i % 1000 == 0:
            print("Read {} images".format(i))

        # if i == 1000:
        #     break

    sizes = np.array(image_sizes)

    plt.figure(dpi=150)
    plt.hist2d(sizes[:, 0], sizes[:, 1], bins=(50, 50), cmap=plt.cm.jet)
    plt.title("Category: {}, num images: {}".format(fig_name,  len(image_names)))
    plt.savefig(os.path.join(OUTPUT_PATH, "image_sizes_{}.png".format(fig_name)))


def GetImageDimensionsCounts(image_names):
    image_size_counts = {}
    for i, image_name in enumerate(image_names):
        # print(image_name)
        img = cv2.imread(os.path.join(VOC_PATH, "JPEGImages", image_name))
        image_size_str = "{}x{}".format(img.shape[0], img.shape[1])

        if image_size_str in image_size_counts:
            image_size_counts[image_size_str] += 1
        else:
            image_size_counts[image_size_str] = 1

        if i > 0 and i % 1000 == 0:
            print("Read {} images".format(i))

    for size, count in image_size_counts.items():
        print("{}\t{}".format(size, count))


if __name__ == '__main__':
    GetImageDimensionsHeatmap(os.listdir(os.path.join(VOC_PATH, "JPEGImages")), "ALL")
    GetImageDimensionsCounts(os.listdir(os.path.join(VOC_PATH, "JPEGImages")))

