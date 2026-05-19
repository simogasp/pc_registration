uv run ./scripts/pc_annotation.py --out test_output --input ./data/pt_cloison_long13k.ply --geom_id 1 --type plane  --class_id 1 --instance_id 0
uv run ./scripts/pc_annotation.py --out test_output --input ./data/cloison_av.ply --geom_id 2 --type plane  --class_id 1 --instance_id 1
uv run ./scripts/pc_annotation.py --out test_output --input ./data/cloison_pbordureav.ply --type cylinder  --class_id 2 --instance_id 0 --geom_id 3

uv run ./scripts/pc_annotation.py --out test_output --input ./data/cloison_middle.ply --type plane  --class_id 1 --instance_id 2 --geom_id 4
uv run ./scripts/pc_annotation.py --out test_output --input ./data/cloison_middle_pb1.ply --type plane  --class_id 4 --instance_id 0 --geom_id 5
uv run ./scripts/pc_annotation.py --out test_output --input ./data/cloison_middle_pb2.ply --type plane  --class_id 4 --instance_id 1 --geom_id 6

uv run ./scripts/pc_annotation.py --out test_output --input ./data/pont_sup_ceiling.ply --type plane  --class_id 6 --instance_id 0 --geom_id 7
uv run ./scripts/pc_annotation.py --out test_output --input ./data/borde_floor.ply --type plane  --class_id 7 --instance_id 0 --geom_id 8