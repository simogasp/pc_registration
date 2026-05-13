"""Point cloud geometry analysis and manipulation utilities."""

import numpy as np
import open3d as o3d


def rough_scale_point_cloud(pcd: o3d.geometry.PointCloud) -> float:
    """Estimate a rough scale of the point cloud based on its oriented bounding box.

    This function computes the minimal oriented bounding box (OBB) of the input
    point cloud and returns a scale factor that is a power of ten closest to the
    maximum extent of the OBB. This scale can be useful for setting parameters
    in visualization or processing algorithms that depend on the size of the
    point cloud.

    Args:
        pcd: The input point cloud to analyze.

    Returns:
        A scale factor that is a power of ten closest to the maximum extent
        of the OBB. For example, if the maximum extent is 3.7 meters, this
        returns 1.0; if it's 45 meters, this returns 10.0.

    Example:
        >>> pcd = o3d.io.read_point_cloud("model.ply")
        >>> scale = rough_scale_point_cloud(pcd)
        >>> print(f"Recommended scale: {scale}")
    """
    obb = pcd.get_minimal_oriented_bounding_box()
    max_extent = max(obb.extent)
    # return the closest power of ten
    return 10 ** np.floor(np.log10(max_extent))


def rough_scale_point_cloud_from_file(pcd_filename: str) -> float:
    """Estimate a rough scale of a point cloud from file.

    This function loads a point cloud from the specified file, computes its
    minimal oriented bounding box (OBB), and returns a scale factor that is
    a power of ten closest to the maximum extent of the OBB. This is a
    convenience wrapper around :func:`rough_scale_point_cloud`.

    Args:
        pcd_filename: The file path to the point cloud to analyze. Supported
            formats include PLY, PCD, XYZ, etc. (depends on Open3D support).

    Returns:
        A scale factor that is a power of ten closest to the maximum extent
        of the OBB.

    Raises:
        RuntimeError: If the point cloud file cannot be read or is empty.

    Example:
        >>> scale = rough_scale_point_cloud_from_file("data/model.ply")
        >>> print(f"Point cloud scale: {scale}")
    """
    pcd = o3d.io.read_point_cloud(pcd_filename)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to read point cloud from {pcd_filename}")
    return rough_scale_point_cloud(pcd)


def align_centers(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> np.ndarray:
    r"""Compute a transformation to align the centroids of two point clouds.

    This function applies optional correction and initial transformations to 
    both point clouds, computes their centroids, and calculates a translation 
    transformation that aligns the centroid of the source point cloud to that 
    of the target point cloud.

    Args:
        source: The source point cloud to be aligned.
        target: The target point cloud (reference).
        trans_init: Initial transformation matrix (4x4) to apply to the source 
            cloud before computing centroids. Default is identity matrix.
        correction: Correction transformation matrix (4x4) to apply to both 
            clouds, typically used to align to a visual reference frame or 
            coordinate system. Default is identity matrix.

    Returns:
        A 4x4 transformation matrix representing the translation that aligns 
        the source centroid to the target centroid. This matrix has the form:
        
        .. math::
        
            T = \\begin{bmatrix}
                1 & 0 & 0 & t_x \\\\
                0 & 1 & 0 & t_y \\\\
                0 & 0 & 1 & t_z \\\\
                0 & 0 & 0 & 1
            \\end{bmatrix}
        
        where :math:`(t_x, t_y, t_z)` is the translation vector.

    Note:
        This function modifies the input point clouds in-place by applying 
        the correction and initial transformations.

    Example:
        >>> source = o3d.io.read_point_cloud("source.ply")
        >>> target = o3d.io.read_point_cloud("target.ply")
        >>> T_align = align_centers(source, target)
        >>> source.transform(T_align)  # Apply alignment
    """
    source.transform(correction)
    target.transform(correction)

    source.transform(trans_init)

    centroid_source = source.get_center()
    centroid_target = target.get_center()

    translation = centroid_target - centroid_source

    transformation = np.eye(4)
    transformation[:3, 3] = translation

    return transformation


def align_centers_from_files(
    source_file: str,
    target_file: str,
    trans_init: np.ndarray = np.identity(4),
    correction: np.ndarray = np.identity(4),
) -> np.ndarray:
    """Compute a transformation to align centroids from point cloud files.

    This function loads the source and target point clouds from the specified
    files and computes a transformation to align their centroids. This is a
    convenience wrapper around :func:`align_centers`.

    Args:
        source_file: File path to the source point cloud.
        target_file: File path to the target point cloud.
        trans_init: Initial transformation matrix (4x4) to apply to the source
            cloud before computing centroids. Default is identity matrix.
        correction: Correction transformation matrix (4x4) to apply to both
            clouds, typically used to align to a visual reference frame or
            coordinate system. Default is identity matrix.

    Returns:
        A 4x4 transformation matrix that aligns the centroids of the source
        and target point clouds.

    Raises:
        RuntimeError: If either point cloud file cannot be read.

    Example:
        >>> T_align = align_centers_from_files(
        ...     "data/source.ply",
        ...     "data/target.ply"
        ... )
        >>> print(f"Translation: {T_align[:3, 3]}")
    """
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    return align_centers(source, target, trans_init=trans_init, correction=correction)
