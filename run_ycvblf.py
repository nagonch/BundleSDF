from run_custom import *
import numpy as np
import trimesh
from eval import compute_pose_errors, get_metrics
import OpenEXR
import quaternion
from scipy.spatial.transform import Rotation as R
from ycbv_lf import YCBV_LF, LFReader
import cv2


def project_frame_to_image(object_to_cam, camera_matrix, image):
    frame_3d = np.array(
        [
            [0, 0, 0, 1],
            [0.1, 0, 0, 1],
            [0, 0.1, 0, 1],
            [0, 0, 0.1, 1],
        ]
    ).T

    frame_in_cam = object_to_cam @ frame_3d

    points_3d = frame_in_cam[:3, :].T

    points_2d = (camera_matrix @ points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]

    img_vis = image.copy()
    origin = tuple(points_2d[0].astype(int))
    x_axis = tuple(points_2d[1].astype(int))
    y_axis = tuple(points_2d[2].astype(int))
    z_axis = tuple(points_2d[3].astype(int))

    cv2.line(img_vis, origin, x_axis, (0, 0, 255), 2)
    cv2.line(img_vis, origin, y_axis, (0, 255, 0), 2)
    cv2.line(img_vis, origin, z_axis, (255, 0, 0), 2)

    return img_vis


def visualize_tracking(rgbs, object_poses, camera_matrix, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    camera_matrix = camera_matrix.cpu().numpy()
    for i, (rgb, pose) in enumerate(zip(rgbs, object_poses)):
        pose = pose.cpu().numpy()
        rgb = rgb.cpu().numpy()
        img_vis = project_frame_to_image(pose, camera_matrix, rgb)
        Image.fromarray(img_vis).save(f"{save_folder}/{str(i).zfill(4)}.png")


def run_ours(video_dir, out_folder, sequence_name, use_gui=False, mesh_scale=1 / 0.4):
    set_seed(0)

    os.system(f"rm -rf {out_folder} && mkdir -p {out_folder}")

    cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml", "r"))
    cfg_bundletrack["SPDLOG"] = int(2)
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_bundletrack["bundle"]["max_BA_frames"] = 10
    cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    cfg_bundletrack["feature_corres"]["map_points"] = True
    cfg_bundletrack["feature_corres"]["resize"] = 400
    cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
    cfg_bundletrack["keyframe"]["min_rot"] = 5
    cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
    cfg_bundletrack["p2p"]["max_dist"] = 0.02
    cfg_bundletrack["p2p"]["max_normal_angle"] = 45
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", "r"))
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=use_gui,
    )
    dataset = YCBV_LF(video_dir, sequence_name)
    for i in range(len(dataset)):
        image_center = dataset[i]["rgb_image"]
        depth_center = dataset[i]["depth_image"]
        # print(depth_center.max(), depth_center.min(), depth_center.mean())
        # raise
        mask_center = dataset[i]["object_mask"]
        object_to_cam = dataset[i]["object_pose"]
        cam_to_world = dataset.camera_poses
        K = np.copy(dataset.camera_matrix.cpu().numpy())
        pose_in_model = np.eye(4)
        id_str = f"{str(i).zfill(4)}"
        tracker.run(
            image_center,
            depth_center,
            K,
            id_str,
            mask=mask_center,
            occ_mask=None,
            pose_in_model=pose_in_model,
        )
        print(i, "done")
    reader = LFReader(dataset)
    tracker.on_finish()

    run_one_video_global_nerf(reader, out_folder=out_folder)


def run_one_video_global_nerf(
    reader, out_folder="/home/bowen/debug/bundlesdf_scan_coffee_415"
):
    set_seed(0)

    out_folder += "/"  #!NOTE there has to be a / in the end

    cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml", "r"))
    cfg_bundletrack["debug_dir"] = out_folder
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml", "r"))
    cfg_nerf["n_step"] = 2000
    cfg_nerf["N_samples"] = 64
    cfg_nerf["N_samples_around_depth"] = 256
    cfg_nerf["first_frame_weight"] = 1
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["finest_res"] = 256
    cfg_nerf["num_levels"] = 16
    cfg_nerf["mesh_resolution"] = 0.002
    cfg_nerf["n_train_image"] = 500
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["frame_features"] = 2
    cfg_nerf["rgb_weight"] = 100

    cfg_nerf["i_img"] = np.inf
    cfg_nerf["i_mesh"] = cfg_nerf["i_img"]
    cfg_nerf["i_nerf_normals"] = cfg_nerf["i_img"]
    cfg_nerf["i_save_ray"] = cfg_nerf["i_img"]

    cfg_nerf["datadir"] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = copy.deepcopy(cfg_nerf["datadir"])

    os.makedirs(cfg_nerf["datadir"], exist_ok=True)

    cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5
    )
    tracker.cfg_nerf = cfg_nerf
    tracker.run_global_nerf(reader=reader, get_texture=True, tex_res=512)
    tracker.on_finish()

    print(f"Done")


if __name__ == "__main__":
    dataset_path = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset"
    sequence_name = "bleach_hard_00_03_chaitanya"
    mesh_name = "021_bleach_cleanser"
    mesh_path = f"{dataset_path}/models/{mesh_name}"
    mesh_file_path = f"{mesh_path}/textured.obj"
    out_folder = f"output/{sequence_name}"
    mesh_scale = 1

    dataset = YCBV_LF(dataset_path, sequence_name)
    images = []
    gt_poses = []
    i_start = 0
    print("getting gt")
    for i in range(i_start, i_start + len(dataset)):
        image_center = dataset[i]["rgb_image"]
        depth_center = dataset[i]["depth_image"]
        mask_center = dataset[i]["object_mask"]
        object_to_cam = dataset[i]["object_pose"]
        cam_to_world = dataset.camera_poses
        images.append(image_center)
        gt_poses.append(object_to_cam)
    print("done")
    images = np.stack(images, axis=0)
    gt_poses = np.stack(gt_poses, axis=0)
    run_ours(dataset_path, out_folder, sequence_name, mesh_scale=mesh_scale)
    est_poses = []
    est_poses_path = f"{out_folder}/ob_in_cam"
    for file in list(sorted(os.listdir(est_poses_path))):
        file_path = f"{est_poses_path}/{file}"
        pose = np.loadtxt(file_path)
        est_poses.append(pose)
    est_poses = np.stack(est_poses, axis=0)
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = (
        np.linalg.inv(pose_est_0) @ pose_gt_0
    )  # only evaluate tracking, without pose estiamtion
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)
    gt_poses, est_poses = (
        torch.tensor(gt_poses).float(),
        torch.tensor(est_poses).float(),
    )

    torch.save(gt_poses, f"{out_folder}/gt_poses.pt")
    torch.save(est_poses, f"{out_folder}/est_poses.pt")

    pose_errors = compute_pose_errors(gt_poses, est_poses)
    adds_vals, add_vals, adds_auc, add_auc = get_metrics(
        gt_poses, mesh_file_path, est_poses
    )
    pose_errors.update({"adds_auc": float(adds_auc), "add_auc": float(add_auc)})
    visualize_tracking(
        torch.tensor(images),
        torch.tensor(est_poses),
        torch.tensor(dataset.camera_matrix),
        f"{out_folder}/frame_vis_car",
    )
    with open(f"{out_folder}/metrics_car.yaml", "w") as file:
        yaml.dump(pose_errors, file)
