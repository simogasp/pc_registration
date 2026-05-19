  #!/bin/bash

# Example of commands to run to fuse the original scans into a single point cloud, with different filtering and subsampling options.

uv run ./scripts/sequence_registration/fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_0-204 --filter-distant --distance-percentile 99.999 --start_scan 0 --end_scan 204

uv run ./scripts/sequence_registration/fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_0-204 --filter-distant --distance-percentile 99.999 --start-scan 0 --end-scan 204

uv run ./scripts/sequence_registration/fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_0-204_step10 --filter-distant --distance-percentile 99.999 --start-scan 0 --end-scan 204 --step 10 --skip-raw

uv run ./scripts/sequence_registration/fuse_scans.py --input data/dataset_real_lidar/ --output output/sequence_registration/fuse/filtered_distance_full_step10 --filter-distant --distance-percentile 99.999  --step 10 --skip-raw



# new datasets

### acq1-playground-2026-04-20 - extracted
uv run ./scripts/sequence_registration/fuse_scans.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm --output output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/filtered_distance_full --filter-distant --distance-percentile 99.999  --skip-raw

uv run ./scripts/load_and_display.py --input output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/filtered_distance_full/fused_map.ply

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/ --speed 0.05 --fused-map output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/filtered_distance_full/fused_map.ply


### acq1-playground-2026-04-20 - extracted_1
uv run ./scripts/sequence_registration/fuse_scans.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm --output output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/filtered_distance_full --filter-distant --distance-percentile 99.999  --skip-raw

uv run ./scripts/load_and_display.py --input output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/filtered_distance_full/fused_map.ply

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/ --speed 0.05 --fused-map output/sequence_registration/fuse/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/filtered_distance_full/fused_map.ply


### acq2-A-lateral-panel-2026-04-28
uv run ./scripts/sequence_registration/fuse_scans.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm --output output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full --filter-distant --distance-percentile 99.999  --skip-raw

uv run ./scripts/load_and_display.py --input output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full/fused_map.ply

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/ --speed 0.05 --fused-map output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full/fused_map.ply










uv run ./scripts/sequence_registration/fuse_scans.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm --output output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full --filter-distant --distance-percentile 99.999  --skip-raw

uv run ./scripts/load_and_display.py --input output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full/fused_map.ply

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/ --speed 0.05 --fused-map output/sequence_registration/fuse/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/filtered_distance_full/fused_map.ply