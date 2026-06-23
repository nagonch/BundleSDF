"""FPS + GPU benchmark for BundleSDF on a single prod sequence.

Measures per-frame tracking throughput for frames 1..N (frame 0 is used as
warm-up / tracker initialisation, matching the ICP benchmark convention).
IO is excluded: all frames are pre-loaded into RAM before timing starts.
torch.cuda.synchronize() fences each frame so reported times reflect real GPU
work, not just Python dispatch latency.

on_finish() (global NeRF refinement) is intentionally excluded from the FPS
measurement — it is a once-per-sequence postprocess, not per-frame cost.

Run from the BundleSDF repo root:
    python bench/fps_bundlesdf.py
    python bench/fps_bundlesdf.py --seq mustard0 --depth synth
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import yaml as pyyaml

# Allow imports from BundleSDF root regardless of cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# GpuMonitor lives in the sibling ReLiFT-6DoF repo.
RELIFT_BENCH = os.path.join(
    os.path.dirname(REPO_ROOT), "ReLiFT-6DoF", "bench"
)
sys.path.insert(0, os.path.dirname(RELIFT_BENCH))
sys.path.insert(0, RELIFT_BENCH)
from gpu_monitor import GpuMonitor

from bundlesdf import BundleSdf, set_seed
from ycbv_lf import YCBV_LF_Prod, PROD_SEQUENCE_TO_OBJECT

DATASET_ROOT = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new"
MESH_ROOT = f"{DATASET_ROOT}/object_meshes_reconstructed"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_tracker(out_folder: str) -> BundleSdf:
    """Replicate the config setup from run_bundlesdf() and return a tracker."""
    cfg_bundletrack = pyyaml.load(
        open(f"{REPO_ROOT}/BundleTrack/config_ycbvlf_prod.yml", "r"),
        Loader=pyyaml.FullLoader,
    )
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    pyyaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = pyyaml.load(
        open(f"{REPO_ROOT}/config.yml", "r"), Loader=pyyaml.FullLoader
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

    return BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="BundleSDF FPS/GPU benchmark")
    ap.add_argument("--split", default="objects")
    ap.add_argument("--refl", default="1.0", help="reflectivity (default 1.0)")
    ap.add_argument("--seq", default="bleach0")
    ap.add_argument("--depth", default="gt", choices=["gt", "synth"])
    args = ap.parse_args()

    seq_dir = os.path.join(DATASET_ROOT, f"{args.split}_{args.refl}", args.seq)
    if not os.path.isdir(seq_dir):
        sys.exit(f"sequence not found: {seq_dir}")

    object_name = PROD_SEQUENCE_TO_OBJECT[args.seq]
    depth_tag = "dgt" if args.depth == "gt" else "dsim"
    mesh_path = f"{MESH_ROOT}/{object_name}_r{args.refl}_{depth_tag}/model.obj"
    if not os.path.exists(mesh_path):
        sys.exit(f"mesh not found: {mesh_path}")

    print(
        f"BundleSDF | {args.split}_{args.refl}/{args.seq} | depth={args.depth}"
    )

    # --- dataset & pre-load (IO excluded from timing) --------------------
    dataset = YCBV_LF_Prod(seq_dir, mesh_path, depth_mode=args.depth)
    K = dataset.camera_matrix.cpu().numpy().astype(np.float64)
    n = len(dataset)
    print(f"  pre-loading {n} frames …")
    frames = [dataset[i] for i in range(n)]

    # --- tracker setup (not timed) ----------------------------------------
    set_seed(0)
    out_folder = tempfile.mkdtemp(prefix="bsdf_bench_")
    try:
        tracker = _build_tracker(out_folder)

        # Warm-up: run frame 0 through the full tracker pipeline.
        # This initialises CUDA kernels and the tracker's internal state.
        print("  warm-up (frame 0) …")
        f0 = frames[0]
        tracker.run(
            f0["rgb_image"],
            f0["depth_image"],
            K,
            "0000",
            mask=f0["object_mask"],
            occ_mask=None,
            pose_in_model=np.eye(4),
        )
        _sync()

        # --- timed loop (frames 1 .. N-1) ---------------------------------
        times: list[float] = []
        with GpuMonitor() as mon:
            for i in range(1, n):
                fi = frames[i]
                _sync()
                t0 = time.perf_counter()
                tracker.run(
                    fi["rgb_image"],
                    fi["depth_image"],
                    K,
                    str(i).zfill(4),
                    mask=fi["object_mask"],
                    occ_mask=None,
                    pose_in_model=np.eye(4),
                )
                _sync()
                times.append(time.perf_counter() - t0)

        # Clean up tracker background processes (not timed).
        print("  on_finish() …")
        tracker.on_finish()

    finally:
        shutil.rmtree(out_folder, ignore_errors=True)

    # --- report -----------------------------------------------------------
    label = "bundlesdf"
    header = (
        f"BundleSDF | {args.split}_{args.refl}/{args.seq} | depth={args.depth}"
    )
    lines = [header]
    if times:
        tot = sum(times)
        nt = len(times)
        lines += [
            f"── FPS [{label}] ──",
            f"  {nt / tot:6.2f} FPS   "
            f"({nt} timed frames, {1000.0 * tot / nt:.1f} ms/frame mean)",
        ]
    lines.append(mon.format_report(label, torch_module=torch))

    report = "\n".join(lines)
    out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fps_bundlesdf.txt")
    with open(out_txt, "w") as f:
        f.write(report + "\n")
    print("\n" + report + f"\n\nsaved -> {out_txt}")


if __name__ == "__main__":
    main()
