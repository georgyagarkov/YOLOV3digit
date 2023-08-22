import config
import numpy as np
import os
import pandas as pd
import torch
from typing import Tuple
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_tensor_from_grid(grid_file):
    current_x = None
    current_column = 0
    tensor = []

    for line_num, line in enumerate(grid_file):
        x_coordinate, y_coordinate, x_amplitude, y_amplitude, z_amplitude = map(float, line.strip().split(' '))
        if current_x is None:
            current_x = x_coordinate
            tensor.append([[x_amplitude, y_amplitude, z_amplitude]])
            continue

        if current_x == x_coordinate:
            tensor[current_column].append([x_amplitude, y_amplitude, z_amplitude])

        if current_x is not None and current_x != x_coordinate:
            tensor.append([[x_amplitude, y_amplitude, z_amplitude]])
            current_column += 1
            current_x = x_coordinate

    tensor = torch.tensor(tensor).permute(2, 0, 1)
    
    return tensor



class YOLODataset(Dataset):
    def __init__(
        self,
        data_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.data_dir = data_dir
        self.sample_paths = []
        for sample_name in os.listdir(data_dir):
            grid_path = os.path.join(data_dir, sample_name, "Grid.txt")
            target_path = os.path.join(data_dir, sample_name, "Target.txt")
            self.sample_paths.append((grid_path, target_path))

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, index):
        img_path, label_path = self.sample_paths[index]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        with open(img_path) as f:
            image = get_tensor_from_grid(f)
            ts = torch.zeros(3,self.image_size,self.image_size)
            """
            for i in range(3):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        ts[i][j][k]=image[i][j][k]
            """
        ts[...,:image.shape[1],:image.shape[2]] = image[...,:image.shape[1],:image.shape[2]]
        for bb in bboxes:
            bb[1]=bb[1]*image.shape[1]/self.image_size
            bb[2]=bb[2]*image.shape[2]/self.image_size
            bb[3]=bb[3]*image.shape[1]/self.image_size
            bb[4]=bb[4]*image.shape[2]/self.image_size
        image = ts
        """
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        """
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)
