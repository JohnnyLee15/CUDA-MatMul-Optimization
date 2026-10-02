from .config import Config
from .constants import FLOATS_PER_FLOAT4, THREADS_PER_WARP


MAX_BLOCK_SIZE = 1024


def is_config_valid(config: Config) -> bool:
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
        config.rows_per_tile_thread <= 0 or
        config.warp_tile_m % config.rows_per_tile_thread != 0 or
        config.rows_per_tile_thread % FLOATS_PER_FLOAT4 != 0
    ):
        return False

    if (
        config.cols_per_tile_thread <= 0 or
        config.warp_tile_n % config.cols_per_tile_thread != 0 or
        config.cols_per_tile_thread % FLOATS_PER_FLOAT4 != 0
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
        (config.warp_tile_m // config.rows_per_tile_thread) *
        (config.warp_tile_n // config.cols_per_tile_thread) !=
        THREADS_PER_WARP
    ):
        return False

    if config.acc_loop_unroll <= 0:
        return False

    return True
