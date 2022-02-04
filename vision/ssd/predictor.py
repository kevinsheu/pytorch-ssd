import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def format_image_features(self, image_features, mask):
        indexes = torch.nonzero(mask)
        selected_features = []
        for index in indexes[:, 0]:
            count = 0
            running_sum = image_features[0].shape[0]
            while index > running_sum:
                count += 1
                running_sum += image_features[count].shape[0]
            offset = index - (running_sum - image_features[count].shape[0])
            features = image_features[count][offset, :]
            selected_features.append(features)

        return selected_features

    def predict(self, image, top_k=-1, prob_threshold=None, output_image_features=False):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes, image_features = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        image_features = [x[0].to(cpu_device) for x in image_features]
        picked_box_probs = []
        picked_labels = []
        picked_features = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            selected_image_features = self.format_image_features(image_features, mask)
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs, selected_image_features = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size,
                                      images_features=selected_image_features)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
            picked_features.extend(selected_image_features)
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        if output_image_features:
            return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4], picked_features
        else:
            return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]