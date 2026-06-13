from bundlesdf import *
import copy
import os, sys, argparse, shutil
import numpy as np
import yaml as pyyaml

from ycbv_lf import YCBV_LF_Prod, PROD_SEQUENCE_TO_OBJECT

code_dir = os.path.dirname(os.path.realpath(__file__))

DATASET_ROOT = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new"
MESH_ROOT = f"{DATASET_ROOT}/object_meshes_reconstructed"
OUTPUT_ROOT = "/home/ngoncharov/cvpr2026/ReLiFT-6DoF/baselines/results_bsdf"
TRACK_ROOT = "/home/ngoncharov/cvpr2026/BundleSDF/output_prod"

REFLECTIVITIES = ["0.0", "0.5", "0.7", "1.0"]
DEPTH_MODES = ["gt", "synth"]


def depth_tag(depth_mode: str) -> str:
    return "dgt" if depth_mode == "gt" else "dsim"


def mesh_path_for(object_name: str, reflectivity: str, depth_mode: str) -> str:
    return (
        f"{MESH_ROOT}/{object_name}_r{reflectivity}_{depth_tag(depth_mode)}/model.obj"
    )


def run_bundlesdf(dataset, out_folder: str):
    set_seed(0)
    # wipe any partial output from a previous crashed run so stale pose
    # files in ob_in_cam/ can't contaminate collect_poses
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    cfg_bundletrack = pyyaml.load(
        open(f"{code_dir}/BundleTrack/config_ycbvlf_prod.yml", "r"),
        Loader=pyyaml.FullLoader,
    )
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    pyyaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = pyyaml.load(
        open(f"{code_dir}/config.yml", "r"), Loader=pyyaml.FullLoader
    )
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    pyyaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=False,
    )

    K = dataset.camera_matrix.cpu().numpy().astype(np.float64)
    for i in range(len(dataset)):
        frame = dataset[i]
        tracker.run(
            frame["rgb_image"],
            frame["depth_image"],
            K,
            str(i).zfill(4),
            mask=frame["object_mask"],
            occ_mask=None,
            pose_in_model=np.eye(4),
        )

    tracker.on_finish()


def collect_poses(out_folder: str):
    ob_in_cam_dir = f"{out_folder}/ob_in_cam"
    files = sorted(os.listdir(ob_in_cam_dir))
    return [np.loadtxt(f"{ob_in_cam_dir}/{f}") for f in files]


def run_sequences(prefix: str, reflectivity: str, depth_mode: str):
    split_dir = f"{DATASET_ROOT}/{prefix}_{reflectivity}"
    sequences = sorted(
        s
        for s in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, s)) and s != "models"
    )

    out_dir = f"{OUTPUT_ROOT}/{depth_mode}/{prefix}_{reflectivity}"
    track_dir = f"{TRACK_ROOT}/{depth_mode}/{prefix}_{reflectivity}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(track_dir, exist_ok=True)

    for seq_name in sequences:
        out_path = f"{out_dir}/{seq_name}.npy"
        if os.path.exists(out_path):
            print(f"[skip] {prefix}_{reflectivity} / {depth_mode} / {seq_name}")
            continue

        object_name = "cube" if prefix == "cube" else PROD_SEQUENCE_TO_OBJECT[seq_name]
        mesh_path = mesh_path_for(object_name, reflectivity, depth_mode)

        if not os.path.exists(mesh_path):
            print(f"[warn] mesh not found: {mesh_path}")
            continue

        seq_path = os.path.join(split_dir, seq_name)
        seq_out_folder = f"{track_dir}/{seq_name}"
        print(
            f"[run]  {prefix}_{reflectivity} / {depth_mode} / {seq_name}  mesh={object_name}"
        )

        try:
            dataset = YCBV_LF_Prod(seq_path, mesh_path, depth_mode=depth_mode)
            run_bundlesdf(dataset, seq_out_folder)
            poses = collect_poses(seq_out_folder)
        except Exception as e:
            print(f"[fail] {prefix}_{reflectivity} / {depth_mode} / {seq_name}: {e}")
            continue

        if not poses:
            print(f"[warn] no poses collected for {seq_name}, skipping")
            continue

        gt_poses = [dataset[i]["object_pose"] for i in range(len(dataset))]

        try:
            est_to_gt = np.linalg.inv(poses[0]) @ gt_poses[0]
        except np.linalg.LinAlgError:
            print(f"[warn] singular pose[0], skipping {seq_name}")
            continue

        poses = [p @ est_to_gt for p in poses]
        np.save(out_path, np.array(poses, dtype=np.float32))
        print(f"[done] saved {out_path}  ({len(poses)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["cube", "objects", "all"],
        default="all",
        help="Which sequence group to run",
    )
    args = parser.parse_args()

    prefixes = {"cube": ["cube"], "objects": ["objects"], "all": ["cube", "objects"]}[
        args.mode
    ]

    for prefix in prefixes:
        for reflectivity in REFLECTIVITIES:
            for depth_mode in DEPTH_MODES:
                run_sequences(prefix, reflectivity, depth_mode)


if __name__ == "__main__":
    main()
