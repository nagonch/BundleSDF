import torch
import os
import json
import numpy as np

sequence_names = [
    "bleach_hard_00_03_chaitanya",
    "bleach0",
    "cracker_box_reorient",
    "cracker_box_yalehand0",
    "mustard_easy_00_02",
    "mustard0",
    "sugar_box_yalehand0",
    "sugar_box1",
    "tomato_soup_can_yalehand",
]
model_names = [
    "021_bleach_cleanser",
    "021_bleach_cleanser",
    "024_cracker_box",
    "024_cracker_box",
    "037_mustard_bottle",
    "037_mustard_bottle",
    "035_sugar_box",
    "035_sugar_box",
    "048_tomato_soup_can",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}


class YCBV_LF:
    def __init__(self, dataset_path, sequence_name):
        assert os.path.exists(
            dataset_path
        ), f"Dataset path {dataset_path} does not exist."
        assert os.path.exists(
            os.path.join(dataset_path, sequence_name)
        ), f"Sequence {sequence_name} does not exist in dataset path {dataset_path}."
        self.dataset_path = dataset_path
        self.sequence_name = sequence_name
        self.sequence_path = os.path.join(dataset_path, sequence_name)
        self.model_name = sequence_to_model[sequence_name]
        assert os.path.exists(
            os.path.join(dataset_path, "models", self.model_name)
        ), f"Model {self.model_name} does not exist in dataset path {dataset_path}."

        self.camera_poses_paths = [
            os.path.join(self.sequence_path, "camera_poses", item)
            for item in list(
                sorted(os.listdir(os.path.join(self.sequence_path, "camera_poses")))
            )
        ]
        with open(os.path.join(self.sequence_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            self.n_views = self.metadata["n_views"]
            self.baseline = self.metadata["x_spacing"]
        self.camera_poses = (
            torch.stack(
                [torch.tensor(np.loadtxt(path)) for path in self.camera_poses_paths]
            )
            .cuda()
            .float()
            .reshape(*self.n_views, 4, 4)
        )

    def __len__(self):
        # Placeholder: return the number of items in the dataset
        return 1000

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    DATASET_PATH = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset"
    SEQUENCE_NAME = "bleach_hard_00_03_chaitanya"
    dataset = YCBV_LF(DATASET_PATH, SEQUENCE_NAME)
    print(dataset.camera_poses.shape)
