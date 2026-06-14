# BundleSDF on YCBV-LF (Production Dataset)

This documents the pipeline for running BundleSDF as a 6-DoF pose tracking baseline on the `prod_dataset_new` light-field dataset, producing `.npy` pose arrays for evaluation in ReLiFT-6DoF.

---

## Files

| File | Role |
|---|---|
| `ycbv_lf.py` | Dataset classes and the `PROD_SEQUENCE_TO_OBJECT` map |
| `run_ycbvlf_prod.py` | Top-level runner: sweeps all conditions, runs BundleSDF, saves poses |
| `BundleTrack/config_ycbvlf_prod.yml` | BundleTrack config tuned for this dataset |

---

## Dataset Layout

```
prod_dataset_new/
  {prefix}_{reflectivity}/        # e.g. objects_0.5, cube_1.0
    {sequence_name}/
      metadata.json               # {"n_views": [5, 5], "x_spacing": ...}
      camera_matrix.txt
      camera_poses/
      object_poses/               # one .txt per LF timestep (4x4, object-in-world)
      depth/                      # GT depth (uint16 PNG, mm)
      depth_synth/                # Simulated depth (float32 PNG, mm)
      LF_XXXX/                    # One directory per timestep
        0000.png ... 0024.png     # 5x5 = 25 camera views
        masks/
          0000.png ... 0024.png
  object_meshes_reconstructed/
    {object_name}_r{reflectivity}_{dgt|dsim}/
      model.obj
```

Each LF directory holds a 5×5 grid of cameras. The pipeline uses **only the center view** (`index = n_cameras // 2 = 12`, i.e. `0012.png`) for both RGB and mask.

---

## Dataset Classes (`ycbv_lf.py`)

### `YCBV_LF_Prod` — production class (used by the runner)

```python
dataset = YCBV_LF_Prod(sequence_path, mesh_path, depth_mode="gt")
```

- `depth_mode="gt"` reads from `depth/`; `"synth"` reads from `depth_synth/`.
- `__getitem__` returns `rgb_image`, `object_mask`, `depth_image` (float32, **meters**), `object_pose` (4×4), `frame_id`.
- Depth is divided by 1000 on load (mm → m).
- Index clamping: `min(idx, len-1)` on depth and pose paths guards against off-by-one when the number of LF directories and depth files differ slightly.

### `YCBV_LF` — original class (older dataset format)

Still present for the original `ycbv-eoat-lf/dataset` split (not `prod_dataset_new`). Constructed differently — takes `dataset_path` + `sequence_name` and uses a hardcoded `sequence_to_model` dict mapping to full YCB model names (`021_bleach_cleanser`, etc.). Not used by `run_ycbvlf_prod.py`.

### `PROD_SEQUENCE_TO_OBJECT`

Maps sequence names to the short mesh name used in `object_meshes_reconstructed`:

```python
"bleach0" -> "bleach_cleanser"
"cracker_box_reorient" -> "cracker_box"
# etc.
```

The runner uses this to resolve the right mesh for each sequence. The `cube` prefix is special-cased to always map to `"cube"` without going through this dict.

---

## Runner (`run_ycbvlf_prod.py`)

### Hardcoded paths (change these if moving machines)

```python
DATASET_ROOT = "/home/ngoncharov/cvpr2026/ycbv-eoat-lf/prod_dataset_new"
MESH_ROOT    = f"{DATASET_ROOT}/object_meshes_reconstructed"
OUTPUT_ROOT  = "/home/ngoncharov/cvpr2026/ReLiFT-6DoF/baselines/results_bsdf"
TRACK_ROOT   = "/home/ngoncharov/cvpr2026/BundleSDF/output_prod"
```

### Sweep

The runner sweeps all combinations of:

- `prefix`: `cube`, `objects`
- `reflectivity`: `0.0`, `0.5`, `0.7`, `1.0`
- `depth_mode`: `gt`, `synth`

```bash
python run_ycbvlf_prod.py --mode all      # both cube and objects (default)
python run_ycbvlf_prod.py --mode cube
python run_ycbvlf_prod.py --mode objects
```

### Per-sequence flow

1. **Skip check**: if `{OUTPUT_ROOT}/{depth_mode}/{prefix}_{reflectivity}/{seq_name}.npy` exists, skip.
2. **Mesh check**: warns and skips if the reconstructed mesh for this `(object, reflectivity, depth_mode)` does not exist.
3. **Tracking folder wipe**: `shutil.rmtree` the tracking output directory before each run. This is intentional — stale `ob_in_cam/*.txt` files from a crashed run would otherwise contaminate `collect_poses`.
4. **BundleSDF run**: feeds center-view RGB, mask, depth, and intrinsics frame-by-frame.
5. **Pose collection**: reads all `.txt` files from `{track_dir}/{seq_name}/ob_in_cam/` in sorted order.
6. **Alignment**: transforms estimated poses into the GT coordinate frame using frame-0 as anchor:
   ```python
   est_to_gt = inv(poses[0]) @ gt_poses[0]
   poses = [p @ est_to_gt for p in poses]
   ```
   The saved `.npy` is therefore in the same coordinate frame as GT — **not** raw BundleSDF output.
7. **Save**: `np.save(out_path, np.array(poses, dtype=np.float32))` — shape `(N, 4, 4)`.

### Output locations

| Artifact | Path |
|---|---|
| BundleSDF tracking output | `output_prod/{depth_mode}/{prefix}_{reflectivity}/{seq_name}/` |
| Final aligned pose array | `ReLiFT-6DoF/baselines/results_bsdf/{depth_mode}/{prefix}_{reflectivity}/{seq_name}.npy` |

---

## BundleTrack Config (`config_ycbvlf_prod.yml`)

Tuned for reflective objects and fast hand-held motion. Key departures from defaults:

| Parameter | Value | Reason |
|---|---|---|
| `min_fm_edges_newframe` | 3 | Reflective surfaces produce sparse features; lower threshold prevents dropping frames |
| `max_dist_no_neighbor` | 999 | No distance limit for non-neighbor matches; allows large inter-frame motion |
| `max_normal_no_neighbor` | 180 | Likewise unconstrained for normals |
| `ransac.inlier_dist` | 0.05 | Loose inlier threshold for noisy 3D from reflective surfaces |
| `ransac.max_trans_neighbor` | 0.5 | Wide motion tolerance |
| `ransac.max_rot_deg_neighbor` | 90 | Wide rotation tolerance |
| `feature_corres.max_dist_neighbor` | 0.1 | Generous neighbor distance |
| `min_match_after_ransac` | 3 | Very low floor; keeps sequences alive with poor texture |

### NeRF config (set in `run_bundlesdf`)

```python
cfg_nerf["continual"] = True
cfg_nerf["trunc_start"] = 0.01
cfg_nerf["trunc"] = 0.01          # tight truncation for small objects
cfg_nerf["mesh_resolution"] = 0.005
cfg_nerf["down_scale_ratio"] = 1  # no downscaling
cfg_nerf["fs_sdf"] = 0.1
cfg_nerf["far"] = 2.0             # from zfar in bundletrack config
```

---

## Gotchas

**Tracking folder must be clean before each run.**
The wipe in `run_bundlesdf` is load-bearing. If you bypass it (e.g. by calling `run_bundlesdf` manually), any leftover `ob_in_cam/` files from a prior partial run will appear in `collect_poses` and corrupt the output.

**Resume only works at the sequence boundary.**
The skip check (`out_path.exists()`) is on the final `.npy`. A sequence that died mid-tracking will be re-run from scratch on the next invocation — but that's correct, because the tracking folder gets wiped.

**The reconstructed mesh must match both reflectivity and depth mode.**
Mesh path: `object_meshes_reconstructed/{object_name}_r{reflectivity}_{dgt|dsim}/model.obj`. Using the wrong mesh (wrong reflectivity or wrong depth-mode variant) silently degrades tracking. The runner warns and skips if the mesh is missing, so a missing mesh won't produce silent bad results — but a wrong mesh will.

**Alignment is to frame 0, not a global optimum.**
The `est_to_gt` correction rigidly transforms the whole estimated trajectory so that the first frame matches GT. Drift in later frames is real tracking error, not an artifact of the alignment.

**`YCBV_LF` vs `YCBV_LF_Prod` are not interchangeable.**
`YCBV_LF` expects the older dataset format and a global `models/` subdirectory with full YCB model names. Pointing it at `prod_dataset_new` will fail at the model-directory assert. Always use `YCBV_LF_Prod` for `prod_dataset_new`.

**Depth units are millimetres on disk, metres in memory.**
Both `YCBV_LF_Prod` and `YCBV_LF` divide by 1000 on load. BundleSDF expects metres. Do not divide again elsewhere.

**The cube prefix bypasses `PROD_SEQUENCE_TO_OBJECT`.**
`cube_*` sequences use `object_name = "cube"` unconditionally. All other sequences go through the dict. If you add a new prefix, you need to add a branch or extend the dict.
