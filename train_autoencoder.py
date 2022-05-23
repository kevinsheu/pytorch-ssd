import numpy as np
import cv2
import os
import torch
import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder_models.nets import AutoEncoderNet, ContrastiveLoss

VOC_PATH = "/home/kevin/data/VOC2012/"
EXTRACTED_DATA_PATH = os.path.join(VOC_PATH, "extracted")

class CustomDataset(Dataset):

    def __init__(self, csv_file, data_dir):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        curr_row = self.df.iloc[idx]

        same_df = self.df[(curr_row['image'] == self.df['image']) &
                          (curr_row['cluster_id'] == self.df['cluster_id']) &
                          (curr_row['box_id'] != self.df['box_id'])]
        diff_df = self.df[(curr_row['image'] != self.df['image']) &
                          (curr_row['cluster_id'] != self.df['cluster_id']) &
                          (curr_row['box_id'] != self.df['box_id'])]

        if same_df.shape[0] == 0 or diff_df.shape[0] == 0:
            return [0]
        same_row = same_df.sample().iloc[0]
        diff_row = diff_df.sample().iloc[0]
        # print(same_row)
        # print(diff_row)

        def get_path_from_row(row):
            return os.path.join(EXTRACTED_DATA_PATH, row['image'],
                                "{}_{}.npy".format(row['cluster_id'], row['box_id']))

        cur_arr = np.load(get_path_from_row(curr_row))
        same_arr = np.load(get_path_from_row(same_row))
        diff_arr = np.load(get_path_from_row(diff_row))

        # print(cur_arr.shape, same_arr.shape, diff_arr.shape)

        return {
            'curr': (cur_arr, curr_row['fm_id']),
            'same': (same_arr, same_row['fm_id']),
            'diff': (diff_arr, diff_row['fm_id']),
        }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
dataset = CustomDataset(os.path.join(VOC_PATH, "box_features.csv"), EXTRACTED_DATA_PATH)

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0)
fm_output_sizes= [
    576,
    1280,
    512,
    256,
    256,
    64
]
fm_model_dict = {}
param_list = []
for i in range(0, 6):
    fm_model_dict[i] = AutoEncoderNet(fm_output_sizes[i])
    fm_model_dict[i].to(device)
    param_list.extend(fm_model_dict[i].parameters())
loss_fn = ContrastiveLoss()

optimizer = optim.SGD(param_list, lr=0.001)
optimizer.zero_grad()
# print(fm_model_dict)

batch_size = 128

for i, data, in enumerate(dataloader):
    if len(data) == 1:
        continue

    def forward_input(input_list):
        fv = input_list[0].to(device)
        fm_id = input_list[1].numpy()[0]
        # print(fv.shape, fm_id)
        return fm_model_dict[fm_id](fv)

    curr_embed = forward_input(data['curr'])
    same_embed = forward_input(data['same'])
    diff_embed = forward_input(data['diff'])

    same_loss = loss_fn(curr_embed, same_embed, 1) / batch_size
    diff_loss = loss_fn(curr_embed, diff_embed, 0) / batch_size
    loss = (same_loss + diff_loss) / batch_size / 2
    loss.backward()

    if i % batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        print(i, loss)


    # diff_embed = forward_input(data['diff'])



