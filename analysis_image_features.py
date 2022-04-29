import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



VOC_PATH = "/home/kevin/data/VOC2012/"
OUTPUT_PATH = "/home/kevin/code/output"

def counts_graph(input_df, fig_name):
    # counts = df["fm_id"].value_counts()
    # print(counts)
    fm_ids = np.arange(0, 6)
    fm_counts = []
    for fm_id in fm_ids:
        count = input_df[input_df["fm_id"] == fm_id]["fm_id"].count()
        fm_counts.append(count)

    fm_counts = np.array(fm_counts)
    fig = plt.figure()
    plt.bar(fm_ids, fm_counts)
    plt.title("Feature map counts ({})".format(fig_name))
    plt.xlabel("Feature Map index")
    plt.ylabel("Count")
    fig.savefig(os.path.join(OUTPUT_PATH, "fm_counts_{}.png".format(fig_name)))


def fm_counts_graph(input_df, fig_name):
    cluster_info = input_df.groupby(['image', 'cluster_id']).size().reset_index().rename(columns={0: 'count'})

    # init count dict
    counts = {}
    for i in range(6):
        for j in range(i + 1, 6):
            counts["{}_{}".format(i, j)] = 0

    # print(counts)

    for index, row in cluster_info.iterrows():
        # print(row)
        boxes = input_df[(input_df["image"] == row["image"]) & (input_df["cluster_id"] == row["cluster_id"])]
        # print(boxes)
        included_fms = np.sort(boxes["fm_id"].unique())
        # print(included_fms)
        for fm_i in range(included_fms.shape[0]):
            for fm_j in range(fm_i + 1, included_fms.shape[0]):
                # print(fm_i, fm_j)
                counts["{}_{}".format(included_fms[fm_i], included_fms[fm_j])] += 1
        # print(counts)
        if index % 5000 == 0 and index > 0:
            print("finish {} of {}".format(index, cluster_info.shape[0]))

        # if index > 50:
        #     break


    sorted_counts = {r: counts[r] for r in sorted(counts, key=counts.get, reverse=True)}
    for key, value in sorted_counts.items():
        print("{}\t{}".format(key, value))

    fig = plt.figure(figsize=(10, 6))
    plt.bar(*zip(*sorted_counts.items()))
    plt.title("Feature map overlap counts ({})".format(fig_name))
    plt.xlabel("Feature Map indexes")
    plt.ylabel("Count")
    fig.savefig(os.path.join(OUTPUT_PATH, "fm_overlap_counts_{}.png".format(fig_name)))
    plt.show()


df = pd.read_csv(os.path.join(VOC_PATH, "box_features.csv"), index_col=0)
print(df)

counts_graph(df, "ALL")
counts_graph(df[df["class_id"] == 7], "CAR")

fm_counts_graph(df, "ALL")
fm_counts_graph(df[df["class_id"] == 7], "CAR")

