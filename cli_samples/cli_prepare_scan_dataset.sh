#!/bin/bash

# This script prepares the scan datasets by converting the extracted xyz point clouds to PLY format and scaling them to millimeters and visualizes the sequences.

### acq2-A-lateral-panel-2026-04-28 - extracted
uv run ./scripts/sequence_registration/prepare_scan_dataset.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted/ -o data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/ --scale 1000 --workers 8

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq2-A-lateral-panel-2026-04-28/extracted_ply_mm/ --speed 0.05

### acq1-playground-2026-04-20 - extracted_1 
uv run ./scripts/sequence_registration/prepare_scan_dataset.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_1 -o data/external/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/ --scale 1000 --workers 4

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_1_ply_mm/ --speed 0.05


## acq1-playground-2026-04-20 - extracted
uv run ./scripts/sequence_registration/prepare_scan_dataset.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted/ -o data/external/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/ --scale 1000 --workers 4

uv run ./scripts/sequence_registration/visualize_sequence.py --input data/external/scans_maquette/acq1-playground-2026-04-20/extracted_ply_mm/ --speed 0.05