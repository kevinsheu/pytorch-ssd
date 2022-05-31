import numpy as np
import cv2
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from autoencoder_models.nets import AutoEncoderNet, ContrastiveLoss, EmbddingDataset

VOC_PATH = "/home/kevin/data/VOC2012/"
EXTRACTED_DATA_PATH = os.path.join(VOC_PATH, "extracted")

TRAIN_PATH = "/home/kevin/train/"

TRAIN_NUM_DIFF_SAMPLE = 9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = EmbddingDataset(os.path.join(VOC_PATH, "box_features.csv"), EXTRACTED_DATA_PATH, split_name="train", get_diff_scale=True,
                                num_diff_sample=TRAIN_NUM_DIFF_SAMPLE)
test_dataset = EmbddingDataset(os.path.join(VOC_PATH, "box_features.csv"), EXTRACTED_DATA_PATH, split_name="test", get_diff_scale=True,
                               num_diff_sample=1)

fm_output_sizes = [
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
# loss_fn = nn.CosineEmbeddingLoss(margin=0.3)

optimizer = optim.Adam(param_list)
optimizer.zero_grad()
# print(fm_model_dict)

distance_fn = nn.CosineSimilarity()

batch_size = 128
print()
for epoch_i in range(4):
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True, num_workers=0)
    for i, data, in enumerate(train_dataloader):
        if len(data) == 1:
            continue

        for fm_i in range(0, 6):
            fm_model_dict[fm_i].train()

        def forward_input(input_list):
            fv = input_list[0].to(device)
            fm_id = input_list[1].numpy()[0]
            return fm_model_dict[fm_id](fv)

        # print(data['diff'])

        curr_embed = forward_input(data['curr'])
        same_embed = forward_input(data['same'])
        total_loss = loss_fn(curr_embed, same_embed, torch.tensor([0]).to(device))
        # diff_embed = forward_input(data['diff'])

        for data_tuple in data['diff']:
            # print(data_tuple)
            diff_embed = forward_input(data_tuple)
            diff_loss = loss_fn(curr_embed, diff_embed, torch.tensor([1]).to(device))
            total_loss = total_loss + diff_loss

        # diff_loss = loss_fn(curr_embed, diff_embed, torch.tensor([-1]).to(device))
        # total_loss = same_loss + diff_loss
        per_step_loss = total_loss / batch_size / (TRAIN_NUM_DIFF_SAMPLE + 1)
        per_step_loss.backward()


        if i % batch_size == 0 and i > 0:
            optimizer.step()
            optimizer.zero_grad()
            print(i, per_step_loss)

        if i % (batch_size * 7) == 0:
            for fm_i in range(0, 6):
                fm_model_dict[fm_i].eval()

            with torch.no_grad():
                test_dataloader = DataLoader(test_dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
                same_dist_list = []
                diff_dist_list = []
                for test_i, test_data in enumerate(test_dataloader):
                    if len(test_data) == 1:
                        continue
                    if test_i > 1000:
                        break
                    # print(test_data)
                    curr_embed = forward_input(test_data['curr'])
                    same_embed = forward_input(test_data['same'])
                    diff_embed = forward_input(test_data['diff'][0])
                    same_dist = F.pairwise_distance(curr_embed, same_embed).to('cpu').numpy()[0]
                    diff_dist = F.pairwise_distance(curr_embed, diff_embed).to('cpu').numpy()[0]

                    same_dist_list.append(same_dist)
                    diff_dist_list.append(diff_dist)

                # print(same_dist_list)
                # print(diff_dist_list)
                same_dist_list = np.array(same_dist_list)
                diff_dist_list = np.array(diff_dist_list)
                sum_correct = np.sum(same_dist_list > 0)
                diff_correct = np.sum(diff_dist_list < 0)
                sum_avg = np.average(same_dist_list)
                diff_avg = np.average(diff_dist_list)
                # print("same: {}/{},avg {}\t diff: {}/{}, avg {}".format(sum_correct, same_dist_list.shape[0], sum_avg,
                #                                                          diff_correct, diff_dist_list.shape[0], diff_avg))
                print("same avg: {}, diff avg: {}".format(sum_avg, diff_avg))

    # if i > 10000:
    #     break


for fm_id, model in fm_model_dict.items():
    print(fm_id, model)
    torch.save(model.state_dict(), os.path.join(TRAIN_PATH, "fm_{}.pt".format(fm_id)))



