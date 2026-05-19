#!/bin/bash


# # call it in a loop for 2 to 60 changing the target
# for i in $(seq 8 60); do
#     echo "Registering 1.ply to ${i}.ply"
#     uv run ./scripts/global_registration.py --target data/external/nio/new/dataset_4/1.ply --source data/external/nio/new/dataset_4/${i}.ply --max_iter_icp 100000 --voxel-size 0.420 --min-fitness 0.5 --refinement-voxel-size 0.150 --use-gicp
# done

for i in $(seq 1 60); do
    echo "Registering 1.ply to ${i}.ply"
    uv run ./scripts/global_registration.py --target  data/external/nio/new/dataset_4/map/map4.ply --source data/external/nio/new/dataset_3/${i}.ply --max_iter_icp 100000 --voxel-size 0.320 --min-fitness 0.94 --refinement-voxel-size 0.150 --use-gicp
done