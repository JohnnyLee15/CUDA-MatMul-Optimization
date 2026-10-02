import subprocess
from pathlib import Path

from .constants import OBJECT_FILE_EXT, ROOT_DIR, OBJECTS_DIR


def compile_kernel(kernel_path: Path) -> Path:
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    object_path = OBJECTS_DIR / f"{kernel_path.stem}.{OBJECT_FILE_EXT}"
    command = [
        "nvcc",
        "-std=c++20",
        "-O3",
        "-arch=sm_86",
        "-I", f"{ROOT_DIR}/include",
        "-c", str(kernel_path),
        "-o", str(object_path),
    ]

    subprocess.run(command, check=True)

    return object_path
