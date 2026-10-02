from .config import Config, MatrixSizes
from .constants import FLOATS_PER_FLOAT4, THREADS_PER_WARP


MAX_BLOCK_SIZE = 1024
BYTES_PER_FLOAT = 4
MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK = 48 * 1024


def is_config_valid(config: Config, matrix_sizes: MatrixSizes) -> bool:
    if config.block_y <= 0 or config.block_x <= 0:
        return False

    if (
        config.num_threads > MAX_BLOCK_SIZE or
        config.num_threads % THREADS_PER_WARP != 0
    ):
        return False

    if (
        config.warp_tile_n <= 0 or
        config.tile_n % config.warp_tile_n != 0
    ):
        return False

    if (
        config.warp_tile_m <= 0 or
        config.tile_m % config.warp_tile_m != 0
    ):
        return False

    if (
        config.rows_per_thread_tile <= 0 or
        config.warp_tile_m % config.rows_per_thread_tile != 0 or
        config.rows_per_thread_tile % FLOATS_PER_FLOAT4 != 0
    ):
        return False

    if (
        config.cols_per_thread_tile <= 0 or
        config.warp_tile_n % config.cols_per_thread_tile != 0 or
        config.cols_per_thread_tile % FLOATS_PER_FLOAT4 != 0
    ):
        return False

    if config.floats_per_tile_load_pass % config.tile_n != 0:
        return False


    if (
        config.tile_k <= 0 or
        config.tile_k % FLOATS_PER_FLOAT4 != 0 or
        config.floats_per_tile_load_pass % config.tile_k != 0 or
        config.tile_k % config.b_tile_row_stride != 0 or
        config.tile_m % config.a_tile_row_stride != 0
    ):
        return False

    if (
        (config.warp_tile_m // config.rows_per_thread_tile) *
        (config.warp_tile_n // config.cols_per_thread_tile) !=
        THREADS_PER_WARP
    ):
        return False

    if (
        matrix_sizes.m % config.tile_m != 0
        or matrix_sizes.n % config.tile_n != 0
        or matrix_sizes.k % config.tile_k != 0
    ):
        return False

    if config.acc_loop_unroll <= 0:
        return False

    shared_memory_bytes = (
        config.tile_k * config.tile_m +
        config.tile_k * config.tile_n
    ) * BYTES_PER_FLOAT

    if shared_memory_bytes > MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK:
        return False

    return True
