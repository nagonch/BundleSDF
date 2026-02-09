import torch
import os
import json
import numpy as np
from PIL import Image
import trimesh
import copy

sequence_names = [
    "bleach_hard_00_03_chaitanya",
    "bleach0",
    "cracker_box_reorient",
    "cracker_box_yalehand0",
    "mustard_easy_00_02",
    "mustard0",
    "sugar_box_yalehand0",
    "sugar_box1",
    "tomato_soup_can_yalehand0",
]
model_names = [
    "021_bleach_cleanser",
    "021_bleach_cleanser",
    "003_cracker_box",
    "003_cracker_box",
    "006_mustard_bottle",
    "006_mustard_bottle",
    "004_sugar_box",
    "004_sugar_box",
    "005_tomato_soup_can",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}


class YCBV_EOAT:
    def __init__(
        self,
        dataset_path,
        sequence_name,
        reference_mesh_path=None,
        keyframes_only=False,
        key_frames_dataset_path=None,
    ):
        assert os.path.exists(
            dataset_path
        ), f"Dataset path {dataset_path} does not exist."
        assert os.path.exists(
            os.path.join(dataset_path, sequence_name)
        ), f"Sequence {sequence_name} does not exist in dataset path {dataset_path}"
        self.dataset_path = dataset_path
        self.sequence_name = sequence_name
        self.sequence_path = os.path.join(dataset_path, sequence_name)
        self.model_name = sequence_to_model[sequence_name]
        assert os.path.exists(
            os.path.join(dataset_path, "models", self.model_name)
        ), f"Model {self.model_name} does not exist in dataset path {dataset_path}"

        self.gt_mesh = trimesh.load(
            f"{dataset_path}/models/{self.model_name}/textured.obj"
        )
        if reference_mesh_path is not None:
            self.mesh = trimesh.load(
                f"{reference_mesh_path}/{self.model_name[4:]}/model.obj"
            )
        else:
            self.mesh = copy.deepcopy(self.gt_mesh)

        self.camera_poses = np.eye(4)
        self.camera_matrix = (
            torch.tensor(np.loadtxt(os.path.join(self.sequence_path, "cam_K.txt")))
            .cuda()
            .float()
        )

        self.depth_dir = os.path.join(self.sequence_path, "depth")
        self.depth_paths = [
            os.path.join(self.depth_dir, item)
            for item in list(sorted(os.listdir(self.depth_dir)))
        ]
        self.object_poses_dir = os.path.join(self.sequence_path, "annotated_poses")
        self.object_poses_paths = [
            os.path.join(self.object_poses_dir, item)
            for item in list(sorted(os.listdir(self.object_poses_dir)))
        ]

        self.rgb_dir = os.path.join(self.sequence_path, "rgb")
        self.rgb_paths = [
            os.path.join(self.rgb_dir, item)
            for item in list(sorted(os.listdir(self.rgb_dir)))
        ]

        self.mask_dir = os.path.join(self.sequence_path, "gt_mask")
        self.mask_paths = [
            os.path.join(self.mask_dir, item)
            for item in list(sorted(os.listdir(self.mask_dir)))
        ]

        if keyframes_only:
            self.keyframes_sequence_path = os.path.join(
                key_frames_dataset_path, sequence_name
            )
            keyframe_indices = [
                int(filename[3:])
                for filename in sorted(os.listdir(self.keyframes_sequence_path))
                if "LF_" in filename
            ]
            self.rgb_paths = [self.rgb_paths[idx] for idx in keyframe_indices]
            self.depth_paths = [self.depth_paths[idx] for idx in keyframe_indices]
            self.mask_paths = [self.mask_paths[idx] for idx in keyframe_indices]
            self.object_poses_paths = [
                self.object_poses_paths[idx] for idx in keyframe_indices
            ]

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]
        mask_path = self.mask_paths[idx]

        object_pose_path = self.object_poses_paths[idx]
        frame_id = int(rgb_path.split("/")[-1].split(".")[0])

        rgb_image = np.array(Image.open(rgb_path).convert("RGB")).astype(np.uint8)
        object_mask = (np.array(Image.open(mask_path)) * 255.0).astype(np.uint8)
        depth_image = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        object_pose = np.loadtxt(object_pose_path)
        return {
            "rgb_image": rgb_image,
            "object_mask": object_mask,
            "depth_image": depth_image,
            "object_pose": object_pose.astype(np.float32),
            "frame_id": frame_id,
        }


class YCBVReader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.K = torch.clone(self.dataset.camera_matrix).cpu().numpy()
        self.id_strs = [str(i).zfill(4) for i in range(len(dataset))]
        self.colors = []
        self.depths = []
        self.masks = []
        for i in range(len(dataset)):
            frame = self.dataset[i]
            self.colors.append(frame["rgb_image"])
            self.depths.append(frame["depth_image"])
            self.masks.append(frame["object_mask"])

    def get_color(self, id):
        # idx = self.id_strs.index(id)
        return self.colors[id]

    def get_depth(self, id):
        # idx = self.id_strs.index(id)
        return self.depths[id]

    def get_mask(self, id):
        # idx = self.id_strs.index(id)
        return self.masks[id]


if __name__ == "__main__":
    DATASET_PATH = "/home/ngoncharov/cvpr2026/datasets/ycb_in_eoat/"
    SEQUENCE_NAME = "bleach0"
    dataset = YCBV_EOAT(DATASET_PATH, SEQUENCE_NAME)
    from utilities import *

    visualizer = Visualizer()
    for i, frame in enumerate(dataset):
        if i == 20:
            break
        camera_matrix = dataset.camera_matrix
        depth = frame["depth_image"]
        points = (
            backproject_depth_to_pointcloud(
                None, torch.tensor(depth).cuda(), torch.tensor(camera_matrix).cuda()
            )
            .cpu()
            .numpy()
        )
        colors = frame["rgb_image"].reshape(-1, 3)
        if "masks" in frame and frame.get("masks", None) is not None:
            mask = frame["object_mask"].reshape(-1)
            points = points.reshape(-1, 3)[mask.reshape(-1) > 0]
            colors = colors.reshape(-1, 3)[mask.reshape(-1) > 0]

        object_to_base_pose = frame["object_pose"]
        visualizer.add_point_cloud(
            f"points_{i}", points, colors=colors, point_size=1e-3
        )
        visualizer.add_frame(name=f"obj_{i}", frame_T=object_to_base_pose)
    visualizer.run()
