import subprocess
from pathlib import Path

from .constants import (
    OBJECT_FILE_EXT,
    ROOT_DIR,
    OBJECTS_DIR,
    EXECUTABLES_DIR,
)


def compile_kernel(kernel_path: Path) -> Path:
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    object_path = OBJECTS_DIR / f"{kernel_path.stem}.{OBJECT_FILE_EXT}"
    command = [
        "nvcc",
        "-std=c++20",
        "-O3",
        "-arch=sm_86",
        "-I", f"{ROOT_DIR}/include",
        "-I", f"{ROOT_DIR}/autotune",
        "-c", str(kernel_path),
        "-o", str(object_path),
    ]

    subprocess.run(command, check=True)

    return object_path


def compile_benchmark_sources() -> list[Path]:
    sources = [
        ROOT_DIR / "autotune" / "benchmark_main.cpp",
        ROOT_DIR / "src" / "matrix.cpp",
        ROOT_DIR / "src" / "validation.cpp",
        ROOT_DIR / "src" / "cudaCheck.cpp",
        ROOT_DIR / "src" / "cuda" / "matCudaUtils.cu",
        ROOT_DIR / "src" / "cuda" / "matMulCudaNaive.cu",
    ]

    benchmark_objects_dir = OBJECTS_DIR / "benchmark"
    benchmark_objects_dir.mkdir(parents=True, exist_ok=True)

    object_paths = []
    for source_path in sources:
        object_path = (benchmark_objects_dir / f"{source_path.stem}.{OBJECT_FILE_EXT}")
        command = [
            "nvcc",
            "-std=c++20",
            "-O3",
            "-arch=sm_86",
            "-I", str(ROOT_DIR / "include"),
            "-I", str(ROOT_DIR / "autotune"),
            "-c", str(source_path),
            "-o", str(object_path),
        ]

        subprocess.run(command, check=True)
        object_paths.append(object_path)

    return object_paths


def link_benchmark(kernel_object_path: Path, benchmark_objects: list[Path]) -> Path:
    EXECUTABLES_DIR.mkdir(parents=True, exist_ok=True)
    executable_path = EXECUTABLES_DIR / kernel_object_path.stem

    command = [
        "nvcc",
        "-std=c++20",
        "-O3",
        "-arch=sm_86",
        "-I", f"{ROOT_DIR}/include",
        "-I", f"{ROOT_DIR}/autotune",
        str(kernel_object_path),
        *[str(path) for path in benchmark_objects],
        "-o", str(executable_path),
    ]

    subprocess.run(command, check=True)

    return executable_path
