from itertools import product
from math import prod
import shutil
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)

from .config import (
    MatrixSizes,
    Config,
    KernelSpec,
    ATileMappingParams,
    BTileMappingParams,
    KernelLayout,
)
from .validator import is_config_valid
from .generator import generate_kernel
from .compiler import compile_kernel, link_benchmark, compile_benchmark_sources
from .runner import run_benchmark
from .bank_conflicts import find_b_tile_mapping_mask, find_a_tile_mapping_mask_and_shift
from .results import write_results
from .constants import (
    THREADS_PER_WARP,
    KERNEL_OBJECTS_DIR,
    BENCHMARK_OBJECTS_DIR,
    EXECUTABLES_DIR,
)


NUM_COMPILE_WORKERS = 16
NUM_MAPPING_WORKERS = 16

RESET_PROGRESS_LINE = "\r\033[2K"

BLOCK_X_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
BLOCK_Y_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)

ROWS_PER_THREAD_TILE_SWEEP = (4, 8, 16)
COLS_PER_THREAD_TILE_SWEEP = (4, 8, 16)

WARP_SUBTILES_PER_WARP_TILE_M_SWEEP = (1, 2, 4)
WARP_SUBTILES_PER_WARP_TILE_N_SWEEP = (1, 2, 4)

TILE_K_SWEEP = (4, 8, 16, 32, 64, 128)
ACC_LOOP_UNROLL_SWEEP = (1, 2, 4, 8, 16, 32, 64)
THREAD_TILES_PER_WARP_SUBTILE_N_SWEEP = (1, 2, 4, 8, 16, 32)

SWEEPS = (
    BLOCK_X_SWEEP,
    BLOCK_Y_SWEEP,
    ROWS_PER_THREAD_TILE_SWEEP,
    COLS_PER_THREAD_TILE_SWEEP,
    WARP_SUBTILES_PER_WARP_TILE_M_SWEEP,
    WARP_SUBTILES_PER_WARP_TILE_N_SWEEP,
    TILE_K_SWEEP,
    ACC_LOOP_UNROLL_SWEEP,
    THREAD_TILES_PER_WARP_SUBTILE_N_SWEEP,
)

KERNELS_PER_BATCH = 1000

CONFIG_CHECK_PROGRESS_INTERVAL = 1000


def _cleanup_batch_generated_files(keep_top_k_kernels: int, results: list[tuple[str, float, Path]]) -> None:
    if KERNEL_OBJECTS_DIR.exists():
        shutil.rmtree(KERNEL_OBJECTS_DIR)

    if EXECUTABLES_DIR.exists():
        shutil.rmtree(EXECUTABLES_DIR)

    for _, _, kernel_path in results[keep_top_k_kernels:]:
        kernel_path.unlink(missing_ok=True)


def _cleanup_benchmark_objects() -> None:
    if BENCHMARK_OBJECTS_DIR.exists():
        shutil.rmtree(BENCHMARK_OBJECTS_DIR)


def _get_valid_configs(matrix_sizes: MatrixSizes) -> list[Config]:
    num_combos = prod(len(sweep) for sweep in SWEEPS)
    configs = []

    for combo_idx, (
        block_x, block_y,
        rows_per_thread_tile, cols_per_thread_tile,
        warp_subtiles_per_warp_tile_m, warp_subtiles_per_warp_tile_n,
        tile_k,
        acc_loop_unroll,
        thread_tiles_per_warp_subtile_n,
    ) in enumerate(product(*SWEEPS)):
        if combo_idx % CONFIG_CHECK_PROGRESS_INTERVAL == 0:
            print(
                f"{RESET_PROGRESS_LINE}"
                f"Checked {combo_idx}/{num_combos} combinations. "
                f"Found {len(configs)} kernel candidates",
                end="",
                flush=True,
            )

        warp_subtile_n = thread_tiles_per_warp_subtile_n * cols_per_thread_tile
        warp_subtile_m = (THREADS_PER_WARP // thread_tiles_per_warp_subtile_n) * rows_per_thread_tile
        warp_tile_n = warp_subtile_n * warp_subtiles_per_warp_tile_n
        warp_tile_m = warp_subtile_m * warp_subtiles_per_warp_tile_m

        layout = KernelLayout(
            block_x, block_y,
            rows_per_thread_tile, cols_per_thread_tile,
            tile_k,
            warp_tile_m ,warp_tile_n,
            warp_subtiles_per_warp_tile_m, warp_subtiles_per_warp_tile_n,
        )
        config = Config(layout, acc_loop_unroll)
        if is_config_valid(config, matrix_sizes):
            configs.append(config)

    print(
        f"{RESET_PROGRESS_LINE}"
        f"Checked all {num_combos} combinations. "
        f"Found {len(configs)} valid configs.",
        flush=True,
    )
    return configs


def _find_layout_mappings(
    layout: KernelLayout,
) -> tuple[ATileMappingParams, BTileMappingParams] | None:
    b_bit_mask = find_b_tile_mapping_mask(layout)
    if b_bit_mask is None:
        return None

    a_mapping_params = find_a_tile_mapping_mask_and_shift(layout)
    if a_mapping_params is None:
        return None

    return ATileMappingParams(*a_mapping_params), BTileMappingParams(b_bit_mask)


def _find_mappings_for_layouts(
    layouts: list[KernelLayout],
) -> dict[KernelLayout, tuple[ATileMappingParams, BTileMappingParams]]:
    if not layouts:
        return {}

    layout_to_mappings = {}
    print(f"Searching mappings for {len(layouts)} unique layouts.", flush=True)

    with ProcessPoolExecutor(max_workers=min(NUM_MAPPING_WORKERS, len(layouts))) as executor:
        future_to_layout = {
            executor.submit(_find_layout_mappings, layout): layout
            for layout in layouts
        }

        try:
            for i, future in enumerate(as_completed(future_to_layout), start=1):
                layout = future_to_layout[future]
                mappings = future.result()

                if mappings is not None:
                    layout_to_mappings[layout] = mappings

                print(
                    f"{RESET_PROGRESS_LINE}"
                    f"Searched {i}/{len(layouts)} layouts. "
                    f"Found mappings for {len(layout_to_mappings)}.",
                    end="",
                    flush=True,
                )
        except BaseException:
            for future in future_to_layout:
                future.cancel()
            raise

    print()
    return layout_to_mappings


def _get_kernel_specs(matrix_sizes: MatrixSizes) -> list[KernelSpec]:
    configs = _get_valid_configs(matrix_sizes)
    if not configs:
        return []

    unique_layouts = list(dict.fromkeys(config.layout for config in configs))
    layout_to_mappings = _find_mappings_for_layouts(unique_layouts)

    kernel_specs = []
    for config in configs:
        mappings = layout_to_mappings.get(config.layout)
        if mappings is None:
            continue

        a_mapping, b_mapping = mappings
        kernel_specs.append(KernelSpec(config, a_mapping, b_mapping))

    print(f"Found {len(kernel_specs)} kernel candidates.", flush=True)
    return kernel_specs


def _build_candidate(kernel_spec: KernelSpec, benchmark_objects: list[Path]) -> tuple[KernelSpec, Path, Path]:
    kernel_path = generate_kernel(kernel_spec)
    kernel_object_path = compile_kernel(kernel_path)
    executable_path = link_benchmark(
        kernel_object_path,
        benchmark_objects,
    )

    return kernel_spec, kernel_path, executable_path


def _build_candidates(kernel_specs: list[KernelSpec], benchmark_objects: list[Path]) -> list[tuple[KernelSpec, Path, Path]]:
    candidates = []
    with ThreadPoolExecutor(max_workers=NUM_COMPILE_WORKERS) as executor:
        futures = [
            executor.submit(_build_candidate, kernel_spec, benchmark_objects)
            for kernel_spec in kernel_specs
        ]

        try:
            for i, future in enumerate(futures, start=1):
                candidate = future.result()
                candidates.append(candidate)
                kernel_spec, _, _ = candidate
                print(f"{RESET_PROGRESS_LINE}Built {i}/{len(kernel_specs)}: {kernel_spec.config.name}", flush=True, end="")
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    print()
    return candidates


def _check_params(matrix_sizes: MatrixSizes, keep_top_k_kernels: int) -> None:
    if min(matrix_sizes.m, matrix_sizes.n, matrix_sizes.k) <= 0:
        raise ValueError("Matrix dimensions must be positive.")

    if keep_top_k_kernels < 0:
        raise ValueError("The number of kernels to keep cannot be negative.")


def _process_batch(
    batch: list[KernelSpec],
    batch_start_idx: int,
    benchmark_objects: list[Path],
    num_kernel_candidates: int,
    matrix_sizes: MatrixSizes,
) -> list[tuple[str, float, Path]]:
    built_candidates = _build_candidates(batch, benchmark_objects)
    benchmark_results = []

    for i, (kernel_spec, kernel_path, executable_path) in enumerate(built_candidates, start=1):
        config = kernel_spec.config
        print(
            f"{RESET_PROGRESS_LINE}Benchmarking {batch_start_idx + i}/{num_kernel_candidates}: {config.name}",
            flush=True,
            end="",
        )

        time = run_benchmark(executable_path, matrix_sizes.m, matrix_sizes.n, matrix_sizes.k)

        if time is None:
            print(f"{RESET_PROGRESS_LINE}Skipping {config.name}: insufficient kernel resources.", flush=True, end="")
            kernel_path.unlink(missing_ok=True)
            continue

        benchmark_results.append((config.name, time, kernel_path))

    print()
    return benchmark_results


def run_sweep(matrix_sizes: MatrixSizes, keep_top_k_kernels: int) -> list[tuple[str, float, Path]]:
    _check_params(matrix_sizes, keep_top_k_kernels)

    kernel_specs = _get_kernel_specs(matrix_sizes)
    if not kernel_specs:
        return []

    benchmark_results = []
    kernel_spec_batches = [
        kernel_specs[i : i + KERNELS_PER_BATCH]
        for i in range(0, len(kernel_specs), KERNELS_PER_BATCH)
    ]
    benchmark_objects = compile_benchmark_sources()

    for batch_idx, batch in enumerate(kernel_spec_batches):
        benchmark_results.extend(
            _process_batch(
                batch=batch,
                batch_start_idx=(batch_idx*KERNELS_PER_BATCH),
                benchmark_objects=benchmark_objects,
                num_kernel_candidates=len(kernel_specs),
                matrix_sizes=matrix_sizes,
            )
        )

        benchmark_results = sorted(benchmark_results, key=lambda x: x[1])
        results_path = write_results(benchmark_results, matrix_sizes)
        print(f"Batch {batch_idx + 1}/{len(kernel_spec_batches)} complete. "f"Results saved to {results_path}", flush=True)
        _cleanup_batch_generated_files(keep_top_k_kernels, benchmark_results)

    _cleanup_benchmark_objects()
    return benchmark_results
