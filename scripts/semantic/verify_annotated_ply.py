#! /usr/bin/env python3
"""Verify that annotated PLY files can be read correctly."""

import sys
from pathlib import Path

import numpy as np


def read_annotated_ply(ply_path: str):
    """Read a PLY file with custom annotation properties.

    Args:
        ply_path: Path to the PLY file.

    Returns:
        Dictionary with points and annotations.
    """
    with open(ply_path, "r") as f:
        # Read header
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a valid PLY file")

        # Parse header
        vertex_count = 0
        properties = []
        in_header = True

        while in_header:
            line = f.readline().strip()

            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
            elif line == "end_header":
                in_header = False

        # Read data
        data = {name: [] for name, _ in properties}

        for _ in range(vertex_count):
            line = f.readline().strip()
            values = line.split()

            for i, (name, prop_type) in enumerate(properties):
                if prop_type in ["float", "double"]:
                    data[name].append(float(values[i]))
                elif prop_type in ["uchar", "ushort", "uint", "int"]:
                    data[name].append(int(values[i]))

        # Convert to numpy arrays
        np_data: dict[str, np.ndarray] = {
            name: np.array(vals) for name, vals in data.items()
        }

    return np_data, properties


def verify_annotations(ply_path: str):
    """Verify that the PLY file has correct annotations.

    Args:
        ply_path: Path to the annotated PLY file.
    """
    print(f"Reading: {ply_path}")

    data, properties = read_annotated_ply(ply_path)

    print(f"\nProperties found: {[name for name, _ in properties]}")
    print(f"Number of points: {len(data['x'])}")

    # Check required annotation properties
    required = ["semantic_class", "semantic_instance", "geom_id"]
    for prop in required:
        if prop not in data:
            print(f"✗ Missing required property: {prop}")
            return False
        print(f"✓ Found property: {prop}")

    # Count annotated points (non-zero)
    sem_class = data["semantic_class"]
    sem_inst = data["semantic_instance"]
    geom_id = data["geom_id"]

    annotated_mask = (sem_class > 0) | (sem_inst > 0) | (geom_id > 0)
    n_annotated = np.sum(annotated_mask)

    print("\nAnnotation statistics:")
    print(
        f"  Annotated points: {n_annotated} / {len(sem_class)} ({100 * n_annotated / len(sem_class):.1f}%)"
    )

    if n_annotated > 0:
        print(f"  Unique classes: {np.unique(sem_class[annotated_mask])}")
        print(f"  Unique instances: {np.unique(sem_inst[annotated_mask])}")
        print(f"  Unique geom_ids: {np.unique(geom_id[annotated_mask])}")

    print("\n✓ Verification successful!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_annotated_ply.py <path_to_annotated.ply>")
        sys.exit(1)

    ply_path = sys.argv[1]

    if not Path(ply_path).exists():
        print(f"Error: File not found: {ply_path}")
        sys.exit(1)

    if not verify_annotations(ply_path):
        print("✗ Verification failed.")
        sys.exit(1)
