#!/bin/bash

# source root directory (where your folders are)
source_root="/home/ngoncharov/cvpr2026/BundleSDF/results_bundlesdf_box_synth_depth_1.0"

# destination directory
destination_dir="./results_bsdf_box_synth_1.0"

# create destination if it doesn't exist
mkdir -p "$destination_dir"

# iterate over subdirectories
for folder in "$source_root"/*/; do
    # remove trailing slash and extract folder name
    folder_name=$(basename "$folder")

    source_file="${folder}est_poses.npy"

    # check if file exists
    if [ -f "$source_file" ]; then
        destination_file="${destination_dir}/${folder_name}.npy"
        cp "$source_file" "$destination_file"
        echo "Copied: $source_file -> $destination_file"
    fi
done