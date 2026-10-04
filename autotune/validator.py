from .config import Config, MatrixSizes
from .constants import FLOATS_PER_FLOAT4, THREADS_PER_WARP


MAX_BLOCK_SIZE = 1024
BYTES_PER_FLOAT = 4
MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK = 48 * 1024


def _meets_basic_param_constraints(config: Config) -> bool:
    if min(
        config.block_x,
        config.block_y,
        config.rows_per_thread_tile,
        config.cols_per_thread_tile,
        config.tile_k,
        config.warp_tile_m,
        config.warp_tile_n,
        config.warp_subtiles_per_warp_tile_m,
        config.warp_subtiles_per_warp_tile_n,
        config.acc_loop_unroll,
    ) <= 0:
        return False

    return (
        config.num_threads <= MAX_BLOCK_SIZE and
        config.num_threads % THREADS_PER_WARP == 0
    )


def _has_valid_warp_and_subtile_layout(config: Config) -> bool:
    if (
        config.tile_m % config.warp_tile_m != 0 or
        config.tile_n % config.warp_tile_n != 0
    ):
        return False

    if (
        config.warp_tile_m % config.warp_subtiles_per_warp_tile_m != 0 or
        config.warp_tile_n % config.warp_subtiles_per_warp_tile_n != 0
    ):
        return False

    if (
        config.warp_subtile_m % config.rows_per_thread_tile != 0 or
        config.warp_subtile_n % config.cols_per_thread_tile != 0
    ):
        return False

    num_warp_thread_y = config.warp_subtile_m // config.rows_per_thread_tile
    num_warp_thread_x = config.warp_subtile_n // config.cols_per_thread_tile

    return num_warp_thread_y * num_warp_thread_x == THREADS_PER_WARP


def _has_valid_float4_tile_loading(config: Config) -> bool:
    if (
        config.rows_per_thread_tile % FLOATS_PER_FLOAT4 != 0 or
        config.cols_per_thread_tile % FLOATS_PER_FLOAT4 != 0 or
        config.tile_k % FLOATS_PER_FLOAT4 != 0
    ):
        return False

    if (
        config.floats_per_tile_load_pass % config.tile_k != 0 or
        config.floats_per_tile_load_pass % config.tile_n != 0
    ):
        return False

    return (
        config.tile_m % config.a_tile_row_stride == 0 and
        config.tile_k % config.b_tile_row_stride == 0
    )


def _matrix_dimensions_fit_complete_tiles(
    config: Config,
    matrix_sizes: MatrixSizes,
) -> bool:
    return (
        matrix_sizes.m % config.tile_m == 0 and
        matrix_sizes.n % config.tile_n == 0 and
        matrix_sizes.k % config.tile_k == 0
    )


def _fits_static_shared_memory_limit(config: Config) -> bool:
    shared_memory_bytes = (
        config.tile_k * config.tile_m +
        config.tile_k * config.tile_n
    ) * BYTES_PER_FLOAT

    return shared_memory_bytes <= MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK


def is_config_valid(config: Config, matrix_sizes: MatrixSizes) -> bool:
    return (
        _meets_basic_param_constraints(config) and
        _has_valid_warp_and_subtile_layout(config) and
        _has_valid_float4_tile_loading(config) and
        _matrix_dimensions_fit_complete_tiles(config, matrix_sizes) and
        _fits_static_shared_memory_limit(config)
    )
