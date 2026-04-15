# Example of commands to run to create the map from the sequence registration results, visualize it, and visualize an animation of the original scans with the fused map. 
# The first part is run with ground truth poses, while the second part is run without ground truth by estimating the poses.

# with ground truth

# creation of the map from the sequence registration results
uv run ./scripts/sequence_registration/multiway_registration.py --input data/dataset_real_lidar/ --output output/sequence_registration/multiway/gt_loop50_v300 --voxel-size-input 300.0  --filter-distant --distance-percentile 99.998 --loop-closure-distance 50

# visualize final map
uv run ./scripts/load_and_display.py --input output/sequence_registration/multiway/gt_loop50_v300/fused_map_optimized.ply 

# visualize animation of the final map with original scans
uv run ./scripts/sequence_registration/visualize_sequence.py --input data/dataset_real_lidar --speed 0.05 --fused-map output/sequence_registration/multiway/gt_loop50_v300/fused_map_optimized.ply 




# No ground truth
# creation of the map from the sequence registration results
uv run ./scripts/sequence_registration/multiway_registration.py --input data/dataset_real_lidar/ --output output/sequence_registration/multiway/no_gt_loop50_v300 --voxel-size-input 300.0  --filter-distant --distance-percentile 99.998 --loop-closure-distance 50 --no-ground-truth

# visualize final map
uv run ./scripts/load_and_display.py --input output/sequence_registration/multiway/no_gt_loop50_v300/fused_map_optimized.ply 

# visualize animation of the final map with original scans (does not work as without the GT the scans are in a different reference frame than the fused map)
uv run ./scripts/sequence_registration/visualize_sequence.py --input data/dataset_real_lidar --speed 0.05 --fused-map output/sequence_registration/multiway/no_gt_loop50_v300/fused_map_optimized.ply --poses output/sequence_registration/multiway/no_gt_loop50_v300/optimized_poses.json