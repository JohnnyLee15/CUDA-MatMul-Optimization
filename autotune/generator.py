from pathlib import Path

from .config import KernelSpec
from .constants import (
    KERNELS_DIR,
    CUDA_FILE_EXT,
    KERNEL_TEMPLATE_PATH,
)

BLOCK_X = "BLOCK_X"
BLOCK_Y = "BLOCK_Y"

ROWS_PER_THREAD_TILE = "ROWS_PER_THREAD_TILE"
COLS_PER_THREAD_TILE = "COLS_PER_THREAD_TILE"

TILE_M = "TILE_M"
TILE_N = "TILE_N"
TILE_K = "TILE_K"

WARP_TILE_M = "WARP_TILE_M"
WARP_TILE_N = "WARP_TILE_N"

WARP_SUBTILES_PER_WARP_TILE_M = "WARP_SUBTILES_PER_WARP_TILE_M"
WARP_SUBTILES_PER_WARP_TILE_N = "WARP_SUBTILES_PER_WARP_TILE_N"

WARP_SUBTILE_M = "WARP_SUBTILE_M"
WARP_SUBTILE_N = "WARP_SUBTILE_N"

WARP_TILES_PER_BLOCK_N = "WARP_TILES_PER_BLOCK_N"
THREAD_TILES_PER_WARP_SUBTILE_N = "THREAD_TILES_PER_WARP_SUBTILE_N"

A_TILE_ROW_STRIDE = "A_TILE_ROW_STRIDE"
B_TILE_ROW_STRIDE = "B_TILE_ROW_STRIDE"

B_BANK_SECTION_BIT_MASK = "B_BANK_SECTION_BIT_MASK"
A_BANK_SECTION_BIT_MASK = "A_BANK_SECTION_BIT_MASK"
A_K_GROUP_BANK_BIT_SHIFT = "A_K_GROUP_BANK_BIT_SHIFT"

ACC_LOOP_UNROLL = "ACC_LOOP_UNROLL"


def _build_params(kernel_spec: KernelSpec) -> dict[str, int]:
    config = kernel_spec.config
    layout = config.layout
    return {
        BLOCK_X: layout.block_x,
        BLOCK_Y: layout.block_y,
        ROWS_PER_THREAD_TILE: layout.rows_per_thread_tile,
        COLS_PER_THREAD_TILE: layout.cols_per_thread_tile,
        TILE_M: layout.tile_m,
        TILE_N: layout.tile_n,
        TILE_K: layout.tile_k,
        WARP_TILE_M: layout.warp_tile_m,
        WARP_TILE_N: layout.warp_tile_n,
        WARP_SUBTILES_PER_WARP_TILE_M: layout.warp_subtiles_per_warp_tile_m,
        WARP_SUBTILES_PER_WARP_TILE_N: layout.warp_subtiles_per_warp_tile_n,
        WARP_SUBTILE_M: layout.warp_subtile_m,
        WARP_SUBTILE_N: layout.warp_subtile_n,
        WARP_TILES_PER_BLOCK_N: layout.warp_tiles_per_block_n,
        THREAD_TILES_PER_WARP_SUBTILE_N: layout.thread_tiles_per_warp_subtile_n,
        A_TILE_ROW_STRIDE: layout.a_tile_row_stride,
        B_TILE_ROW_STRIDE: layout.b_tile_row_stride,
        B_BANK_SECTION_BIT_MASK: kernel_spec.b_tile_mapping.bank_section_bit_mask,
        A_BANK_SECTION_BIT_MASK: kernel_spec.a_tile_mapping.bank_section_bit_mask,
        A_K_GROUP_BANK_BIT_SHIFT: kernel_spec.a_tile_mapping.k_group_bank_bit_shift,
        ACC_LOOP_UNROLL: config.acc_loop_unroll,
    }


def _build_kernel(params: dict[str, int]) -> str:
    template_str = KERNEL_TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, param in params.items():
        template_str = template_str.replace(f"{{{{ {key} }}}}", str(param))

    return template_str


def generate_kernel(kernel_spec: KernelSpec) -> Path:
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    params = _build_params(kernel_spec)
    kernel_str = _build_kernel(params)
    kernel_path = KERNELS_DIR / f"{kernel_spec.config.name}.{CUDA_FILE_EXT}"
    kernel_path.write_text(kernel_str)
    return kernel_path
