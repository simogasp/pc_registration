"""Registration package for point cloud alignment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("registration")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0.dev"
