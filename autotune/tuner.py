from itertools import product
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .config import MatrixSizes, Config
from .validator import is_config_valid
from .generator import generate_kernel
from .compiler import compile_kernel, link_benchmark, compile_benchmark_sources
from .runner import run_benchmark
from .constants import THREADS_PER_WARP, OBJECTS_DIR, EXECUTABLES_DIR


NUM_COMPILE_WORKERS = 16

BLOCK_X_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BLOCK_Y_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)

ROWS_PER_THREAD_TILE_SWEEP = (4, 8, 16)
COLS_PER_THREAD_TILE_SWEEP = (4, 8, 16)

TILE_K_SWEEP = (4, 8, 16, 32, 64, 128)
ACC_LOOP_UNROLL_SWEEP = (1, 2, 4, 8, 16, 32, 64)
THREAD_TILES_PER_WARP_N_SWEEP = (1, 2, 4, 8, 16, 32)


def _cleanup_generated_files(keep_top_k_kernels: int, results: list[tuple[str, float, Path]]) -> None:
    if OBJECTS_DIR.exists():
        shutil.rmtree(OBJECTS_DIR)

    if EXECUTABLES_DIR.exists():
        shutil.rmtree(EXECUTABLES_DIR)

    for _, _, kernel_path in results[keep_top_k_kernels:]:
        kernel_path.unlink(missing_ok=True)


def _get_valid_configs(matrix_sizes: MatrixSizes) -> list[Config]:
    valid_configs = []
    for (
        block_x, block_y,
        rows_per_thread, cols_per_thread,
        tile_k,
        acc_loop_unroll,
        thread_tiles_per_warp_n,
    ) in product(
        BLOCK_X_SWEEP,
        BLOCK_Y_SWEEP,
        ROWS_PER_THREAD_TILE_SWEEP,
        COLS_PER_THREAD_TILE_SWEEP,
        TILE_K_SWEEP,
        ACC_LOOP_UNROLL_SWEEP,
        THREAD_TILES_PER_WARP_N_SWEEP,
    ):
        warp_tile_n = thread_tiles_per_warp_n * cols_per_thread
        warp_tile_m = rows_per_thread * (THREADS_PER_WARP // thread_tiles_per_warp_n)

        config = Config(
            block_x, block_y,
            rows_per_thread, cols_per_thread,
            tile_k,
            warp_tile_m,warp_tile_n,
            acc_loop_unroll
        )

        if is_config_valid(config, matrix_sizes):
            valid_configs.append(config)

    return valid_configs


def _build_candidate(config: Config, benchmark_objects: list[Path]) -> tuple[Config, Path, Path]:
    kernel_path = generate_kernel(config)
    kernel_object_path = compile_kernel(kernel_path)
    executable_path = link_benchmark(
        kernel_object_path,
        benchmark_objects,
    )

    return config, kernel_path, executable_path


def _build_candidates(configs: list[Config]) -> list[tuple[Config, Path, Path]]:
    benchmark_objects = compile_benchmark_sources()
    candidates = []
    with ThreadPoolExecutor(max_workers=NUM_COMPILE_WORKERS) as executor:
        futures = [
            executor.submit(_build_candidate, config, benchmark_objects)
            for config in configs
        ]

        try:
            for i, future in enumerate(futures, start=1):
                candidate = future.result()
                candidates.append(candidate)
                config, _, _ = candidate
                print(f"Built {i}/{len(configs)}: {config.name}", flush=True,)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    return candidates


def run_sweep(matrix_sizes: MatrixSizes, keep_top_k_kernels: int) -> list[tuple[str, float, Path]]:
    if min(matrix_sizes.m, matrix_sizes.n, matrix_sizes.k) <= 0:
        raise ValueError("Matrix dimensions must be positive.")

    if keep_top_k_kernels < 0:
        raise ValueError("The number of kernels to keep cannot be negative.")

    configs = _get_valid_configs(matrix_sizes)
    print(f"Found {len(configs)} valid configs.")
    if not configs:
        return []

    benchmark_results = []
    built_candidates = _build_candidates(configs)

    for i, (config, kernel_path, executable_path) in enumerate(built_candidates, start=1):
        print(f"Benchmarking {i}/{len(configs)}: {config.name}", flush=True)
        time = run_benchmark(executable_path, matrix_sizes.m, matrix_sizes.n, matrix_sizes.k)

        if time is None:
            print(f"Skipping {config.name}: insufficient kernel resources.", flush=True)
            kernel_path.unlink(missing_ok=True)
            continue

        benchmark_results.append((config.name, time, kernel_path))

    benchmark_results = sorted(benchmark_results, key=lambda x: x[1])
    _cleanup_generated_files(keep_top_k_kernels, benchmark_results)
    return benchmark_results
