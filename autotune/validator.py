from .config import Config, MatrixSizes, KernelLayout
from .constants import FLOATS_PER_FLOAT4, THREADS_PER_WARP


MAX_BLOCK_SIZE = 1024
BYTES_PER_FLOAT = 4
MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK = 48 * 1024
MAX_ACCUMULATORS_PER_THREAD = 128


def _meets_basic_param_constraints(config: Config) -> bool:
    layout = config.layout

    if min(
        layout.block_x,
        layout.block_y,
        layout.rows_per_thread_tile,
        layout.cols_per_thread_tile,
        layout.tile_k,
        layout.warp_tile_m,
        layout.warp_tile_n,
        layout.warp_subtiles_per_warp_tile_m,
        layout.warp_subtiles_per_warp_tile_n,
        config.acc_loop_unroll,
    ) <= 0:
        return False

    return (
        layout.num_threads <= MAX_BLOCK_SIZE and
        layout.num_threads % THREADS_PER_WARP == 0
    )


def _has_valid_warp_and_subtile_layout(layout: KernelLayout) -> bool:
    if (
        layout.tile_m % layout.warp_tile_m != 0 or
        layout.tile_n % layout.warp_tile_n != 0
    ):
        return False

    if (
        layout.warp_tile_m % layout.warp_subtiles_per_warp_tile_m != 0 or
        layout.warp_tile_n % layout.warp_subtiles_per_warp_tile_n != 0
    ):
        return False

    if (
        layout.warp_subtile_m % layout.rows_per_thread_tile != 0 or
        layout.warp_subtile_n % layout.cols_per_thread_tile != 0
    ):
        return False

    num_warp_thread_y = layout.warp_subtile_m // layout.rows_per_thread_tile
    num_warp_thread_x = layout.warp_subtile_n // layout.cols_per_thread_tile

    return num_warp_thread_y * num_warp_thread_x == THREADS_PER_WARP


def _has_valid_float4_tile_loading(layout: KernelLayout) -> bool:
    if (
        layout.rows_per_thread_tile % FLOATS_PER_FLOAT4 != 0 or
        layout.cols_per_thread_tile % FLOATS_PER_FLOAT4 != 0 or
        layout.tile_k % FLOATS_PER_FLOAT4 != 0
    ):
        return False

    if (
        layout.floats_per_tile_load_pass % layout.tile_k != 0 or
        layout.floats_per_tile_load_pass % layout.tile_n != 0
    ):
        return False

    return (
        layout.tile_m % layout.a_tile_row_stride == 0 and
        layout.tile_k % layout.b_tile_row_stride == 0
    )


def _matrix_dimensions_fit_complete_tiles(
    layout: KernelLayout,
    matrix_sizes: MatrixSizes,
) -> bool:
    return (
        matrix_sizes.m % layout.tile_m == 0 and
        matrix_sizes.n % layout.tile_n == 0 and
        matrix_sizes.k % layout.tile_k == 0
    )


def _fits_static_shared_memory_limit(layout: KernelLayout) -> bool:
    shared_memory_bytes = (
        layout.tile_k * layout.tile_m +
        layout.tile_k * layout.tile_n
    ) * BYTES_PER_FLOAT

    return shared_memory_bytes <= MAX_STATIC_SHARED_MEMORY_BYTES_PER_BLOCK


def _fits_accumulator_limit(layout: KernelLayout) -> bool:
    num_accumulators = (
        layout.rows_per_thread_tile
        * layout.cols_per_thread_tile
        * layout.warp_subtiles_per_warp_tile_m
        * layout.warp_subtiles_per_warp_tile_n
    )

    return num_accumulators <= MAX_ACCUMULATORS_PER_THREAD


def is_config_valid(config: Config, matrix_sizes: MatrixSizes) -> bool:
    return (
        _meets_basic_param_constraints(config) and
        _has_valid_warp_and_subtile_layout(config.layout) and
        _has_valid_float4_tile_loading(config.layout) and
        _matrix_dimensions_fit_complete_tiles(config.layout, matrix_sizes) and
        _fits_static_shared_memory_limit(config.layout) and
        _fits_accumulator_limit(config.layout)
    )
