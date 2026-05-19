uv run ./scripts/global_registration.py --source data/sameref/y_-0.75m.ply --target  data/sameref/ry_45_degres.ply --max_iter_icp 100000 --voxel-size 20

uv run ./scripts/global_registration.py --source data/sameref/maquette27k.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 20


## to convert in the same reference frame
uv run ./scripts/flip_and_scale_pc.py --input data/maquette12k.ply --output data/sameref/maquette12k.ply --flip z

uv run ./scripts/flip_and_scale_pc.py --input data/y_-0.75m/pcl_out_time104-116000000.ply --output data/sameref/y_-0.75m.ply --scale 1000 --flip nx


# align scan to maquette E Shape
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/E_shape_maq15k.ply --max_iter_icp 100000 --voxel-size 40 --min-fitness 0.35

# align scan to maquette full
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.4
uv run ./scripts/global_registration.py --source data/sameref/ry_0_degres.ply --target  data/sameref/maquette27k.ply --max_iter_icp 100000 --voxel-size 30 --min-fitness 0.53