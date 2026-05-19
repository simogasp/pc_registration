uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/pt_cloison_long13k_mesh_low.ply --geom_id 1 --type plane  --class_id 1 --instance_id 0
uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/cloison_av_mesh_low.ply --geom_id 2 --type plane  --class_id 1 --instance_id 1
uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/cloison_pbordureav_mesh_low.ply --type cylinder  --class_id 2 --instance_id 0 --geom_id 3

uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/cloison_middle_mesh_low.ply --type plane  --class_id 1 --instance_id 2 --geom_id 4
uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/cloison_middle_pb1_mesh_low.ply --type plane  --class_id 4 --instance_id 0 --geom_id 5
uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/cloison_middle_pb2_mesh_low.ply --type plane  --class_id 4 --instance_id 1 --geom_id 6

uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/pont_sup_ceiling_mesh_low.ply --type plane  --class_id 6 --instance_id 0 --geom_id 7
uv run ./scripts/mesh_annotation.py --out test_output_mesh --input ./data/borde_floor_mesh_low.ply --type plane  --class_id 7 --instance_id 0 --geom_id 8