import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os

VOC_PATH = "/home/kevin/data/VOC2012/"
EXTRACTED_DATA_PATH = os.path.join(VOC_PATH, "extracted")

class AutoEncoderNet(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoderNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(512, 512)
        # self.fc4 = nn.Linear(512, 512)
        # self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        output = self.fc6(x)
        return output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find the pairwise distance or eucledian distance of two output feature vectors
        euclidean_distance = F.pairwise_distance(output1, output2)
        # perform contrastive loss calculation with the distance
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class EmbddingDataset(Dataset):

    def __init__(self, csv_file, data_dir, split_name, train_split=0.7, get_diff_scale=False, num_diff_sample=7):
        self.data_dir = data_dir
        self.get_diff_scale = get_diff_scale
        self.num_diff_sample = num_diff_sample

        entire_df = pd.read_csv(csv_file)
        threshold_index = int(train_split * entire_df.shape[0])
        if split_name == "train":
            self.df = entire_df.iloc[:threshold_index]
            print("loading train set with {}".format(self.df.shape[0]))
        elif split_name == "test":
            self.df = entire_df.iloc[threshold_index:]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        curr_row = self.df.iloc[idx]

        if self.get_diff_scale:
            same_df = self.df[(curr_row['image'] == self.df['image']) &
                              (curr_row['cluster_id'] == self.df['cluster_id']) &
                              (curr_row['box_id'] != self.df['box_id']) &
                              (curr_row['fm_id'] != self.df['fm_id'])]
            diff_df = self.df[(curr_row['image'] != self.df['image']) &
                              # (curr_row['cluster_id'] != self.df['cluster_id']) &
                              # (curr_row['box_id'] != self.df['box_id']) &
                              (curr_row['fm_id'] != self.df['fm_id'])]
        else:
            same_df = self.df[(curr_row['image'] == self.df['image']) &
                              (curr_row['cluster_id'] == self.df['cluster_id']) &
                              (curr_row['box_id'] != self.df['box_id'])]
            diff_df = self.df[
                (curr_row['image'] != self.df['image'])
                # (curr_row['cluster_id'] != self.df['cluster_id']) &
                # (curr_row['box_id'] != self.df['box_id'])
            ]

        if same_df.shape[0] == 0 or diff_df.shape[0] == 0:
            return [0]
        same_row = same_df.sample().iloc[0]
        diff_row = diff_df.sample().iloc[0]
        diff_sample = diff_df.sample(n=self.num_diff_sample)
        # print(diff_sample)
        # print(same_row)
        # print(diff_row)

        def get_path_from_row(row):
            return os.path.join(EXTRACTED_DATA_PATH, row['image'],
                                "{}_{}.npy".format(row['cluster_id'], row['box_id']))

        cur_arr = np.load(get_path_from_row(curr_row))
        same_arr = np.load(get_path_from_row(same_row))

        diff_list = []
        for i in range(diff_sample.shape[0]):
            row = diff_sample.iloc[i]
            arr = np.load(get_path_from_row(row))
            diff_list.append((arr, row['fm_id']))


        # diff_arr = np.load(get_path_from_row(diff_row))

        # print(cur_arr.shape, same_arr.shape, diff_arr.shape)

        return {
            'curr': (cur_arr, curr_row['fm_id']),
            'same': (same_arr, same_row['fm_id']),
            # 'diff': (diff_arr, diff_row['fm_id']),
            'diff': diff_list
        }