from pathlib import Path

from .config import Config
from .constants import (
    KERNELS_DIR,
    CUDA_FILE_EXT,
    KERNEL_TEMPLATE_PATH,
)

BLOCK_X = "BLOCK_X"
BLOCK_Y = "BLOCK_Y"
NUM_THREADS = "NUM_THREADS"

ROWS_PER_THREAD = "ROWS_PER_THREAD"
COLS_PER_THREAD = "COLS_PER_THREAD"

TILE_M = "TILE_M"
TILE_N = "TILE_N"
TILE_K = "TILE_K"

WARP_TILE_M = "WARP_TILE_M"
WARP_TILE_N = "WARP_TILE_N"

WARP_TILES_PER_BLOCK_N = "WARP_TILES_PER_BLOCK_N"
THREAD_TILES_PER_WARP_N = "THREAD_TILES_PER_WARP_N"

FLOATS_PER_LOAD_PASS = "FLOATS_PER_LOAD_PASS"

A_TILE_ROW_STRIDE = "A_TILE_ROW_STRIDE"
B_TILE_ROW_STRIDE = "B_TILE_ROW_STRIDE"


def build_params(config: Config) -> dict[str, int]:
    return {
        BLOCK_X: config.block_x,
        BLOCK_Y: config.block_y,
        NUM_THREADS: config.num_threads,
        ROWS_PER_THREAD: config.rows_per_tile_thread,
        COLS_PER_THREAD: config.cols_per_tile_thread,
        TILE_M: config.tile_m,
        TILE_N: config.tile_n,
        TILE_K: config.tile_k,
        WARP_TILE_M: config.warp_tile_m,
        WARP_TILE_N: config.warp_tile_n,
        WARP_TILES_PER_BLOCK_N: config.warp_tiles_per_block_n,
        THREAD_TILES_PER_WARP_N: config.thread_tiles_per_warp_n,
        FLOATS_PER_LOAD_PASS: config.floats_per_tile_load_pass,
        A_TILE_ROW_STRIDE: config.a_tile_row_stride,
        B_TILE_ROW_STRIDE: config.b_tile_row_stride,
    }


def build_kernel(params: dict[str, int]) -> str:
    template_str = KERNEL_TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, param in params.items():
        template_str.replace(f"{{{{ {key} }}}}", param)

    return template_str


def generate_kernel(config: Config) -> Path:
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    params = build_params(config)
    kernel_str = build_kernel(params)
    kernel_path = KERNELS_DIR / f"{config.name}.{CUDA_FILE_EXT}"
    kernel_path.write_text(kernel_str)
    return kernel_path
