#!/usr/bin/env python3
"""Install the minimal CUDA 13.2 toolkit used by this Axis consumer."""

import hashlib
import io
import json
import os
import pathlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEST = pathlib.Path(__file__).resolve().parents[1] / ".cuda"

# NVIDIA's archives express these linker names as symbolic links.  Keep this
# list fixed so the ignored toolkit cannot introduce links into the Rasat tree.
CUDA_ALIASES = (
    ("lib/libcudart.so", "lib/libcudart.so.13.2.51"),
    ("lib/libcudart.so.13", "lib/libcudart.so.13.2.51"),
    ("lib/libcurand.so", "lib/libcurand.so.10.4.2.51"),
    ("lib/libcurand.so.10", "lib/libcurand.so.10.4.2.51"),
    ("nvvm/lib64/libnvvm.so", "nvvm/lib64/libnvvm.so.4.0.0"),
    ("nvvm/lib64/libnvvm.so.4", "nvvm/lib64/libnvvm.so.4.0.0"),
)


def _checked_path(root: pathlib.Path, relative: str) -> pathlib.Path:
    """Return an in-tree path whose existing parents are not symbolic links."""
    part = pathlib.PurePosixPath(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise RuntimeError(f"unsafe CUDA path: {relative}")
    path = root
    for component in part.parts:
        path /= component
        if path.is_symlink():
            raise RuntimeError(f"unexpected CUDA symbolic link: {path.relative_to(root)}")
    return path


def _same_contents(left: pathlib.Path, right: pathlib.Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while left_chunk := lhs.read(1024 * 1024):
            if left_chunk != rhs.read(len(left_chunk)):
                return False
        return not rhs.read(1)


def materialize_cuda_aliases(root: pathlib.Path) -> None:
    """Replace NVIDIA's six linker aliases with independent regular files."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"CUDA root must be a real directory: {root}")

    alias_paths = {pathlib.PurePosixPath(alias) for alias, _ in CUDA_ALIASES}
    for path in root.rglob("*"):
        if path.is_symlink() and path.relative_to(root) not in alias_paths:
            raise RuntimeError(f"unexpected CUDA symbolic link: {path.relative_to(root)}")

    for alias_name, source_name in CUDA_ALIASES:
        # The source and all of its parents must be real in-tree objects.  The
        # alias itself is allowed to be NVIDIA's expected link and is replaced.
        source = _checked_path(root, source_name)
        alias = root / pathlib.PurePosixPath(alias_name)
        _checked_path(root, str(pathlib.PurePosixPath(alias_name).parent))
        if not source.is_file():
            raise RuntimeError(f"missing CUDA alias source: {source_name}")
        if alias.exists() and not alias.is_symlink():
            if not alias.is_file():
                raise RuntimeError(f"CUDA alias is not a regular file: {alias_name}")
            source_stat = source.stat()
            alias_stat = alias.stat()
            if (
                (source_stat.st_dev, source_stat.st_ino)
                != (alias_stat.st_dev, alias_stat.st_ino)
                and _same_contents(alias, source)
            ):
                continue

        alias.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=alias.parent, prefix=f".{alias.name}.", delete=False) as temporary:
                temporary_name = pathlib.Path(temporary.name)
                with source.open("rb") as versioned:
                    shutil.copyfileobj(versioned, temporary)
                os.fchmod(temporary.fileno(), source.stat().st_mode & 0o777)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, alias)
            temporary_name = None
            directory_fd = os.open(alias.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_name is not None:
                temporary_name.unlink(missing_ok=True)

    residual = [path.relative_to(root) for path in root.rglob("*") if path.is_symlink()]
    if residual:
        raise RuntimeError(f"CUDA symbolic links remain after materialization: {residual}")


PACKAGES = (
    "cuda_cudart",
    "cuda_crt",
    "cuda_cccl",
    "cuda_nvcc",
    "cuda_tileiras",
    "libnvvm",
    "libcurand",
)


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This helper supports Linux x86_64 only; use a system CUDA toolkit.")
    with urllib.request.urlopen(BASE + "redistrib_13.2.0.json", timeout=60) as response:
        manifest = json.load(response)
    DEST.mkdir(exist_ok=True)
    for name in PACKAGES:
        package = manifest[name]["linux-x86_64"]
        marker = DEST / ("." + name + ".sha256")
        if marker.is_file() and marker.read_text().strip() == package["sha256"]:
            print(f"Already installed: {name}", flush=True)
            continue
        print(f"Downloading {name}: {int(package['size']) / 1024**2:.2f} MiB", flush=True)
        with urllib.request.urlopen(BASE + package["relative_path"], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != package["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
            for member in archive.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if len(parts) > 1:
                    member.name = str(pathlib.PurePosixPath(*parts[1:]))
                    archive.extract(member, DEST, filter="data")
        marker.write_text(package["sha256"] + "\n")
    # This intentionally runs after marker processing, including when every
    # package was already installed, so it also repairs damaged local aliases.
    materialize_cuda_aliases(DEST)
    print(f"CUDA_TOOLKIT_PATH={DEST}")


if __name__ == "__main__":
    main()
