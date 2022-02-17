import numpy as np
import cv2
import torch




class Tracker:
    def __init__(self, num_frames_keep=5):
        self.num_frames_keep = num_frames_keep
        self.tracks = []
        self.iter = 0

    def process(self, boxes, features):
        if self.iter == 0:
            for box, embedding in zip(boxes, features):
                self.tracks.append((box, embedding, 0)) # bbox, features, last_seen_iter
            self.iter += 1
            return np.arange(0, len(boxes))

        picked_tracks = []
        for box, embedding in zip(boxes, features):
            best_track = -1
            best_distance = 99999999999999999999999999999999999999
            for i, track in enumerate(self.tracks):
                if embedding.shape != track[1].shape:
                    continue
                if i in picked_tracks:
                    continue
                print("comparing {},{} and {},{}".format(box[0], box[1], track[0][0], track[0][1]))
                distance = np.linalg.norm(embedding.detach().numpy() - track[1].detach().numpy())
                print(distance)

                if distance < 4 and distance < best_distance:
                    best_track = i
                    best_distance = distance

            picked_tracks.append(best_track)

        return_ids = []

        for picked_track, box, embedding in zip(picked_tracks, boxes, features):
            if picked_track == -1:
                return_ids.append(len(self.tracks))
                self.tracks.append((box, embedding, self.iter))
            else:
                return_ids.append(picked_track)
                self.tracks[picked_track] = (box, embedding, self.iter)



        self.iter += 1

        return return_ids


