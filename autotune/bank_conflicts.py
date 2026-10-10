from collections.abc import Callable
from functools import cache

from .config import KernelLayout
from .constants import FLOATS_PER_FLOAT4, THREADS_PER_WARP


NUM_BANKS = 32
NUM_THREADS_PER_FLOAT4_PHASE = NUM_BANKS // FLOATS_PER_FLOAT4
FLOAT4_GROUPS_PER_BANK_REPEAT = NUM_BANKS // FLOATS_PER_FLOAT4
CONFLICT_FREE_DEGREE = 1
A_K_GROUP_SHIFT_CANDIDATES = (0, 1, 2, 3)


TileCoordMapping = Callable[[int, int], tuple[int, int]]


def _get_bank_conflict_degree(phase_indices: list[int]) -> int:
    banks_unique_access_counts = dict.fromkeys(range(NUM_BANKS), 0)
    for idx in set(phase_indices):
        banks_unique_access_counts[idx % NUM_BANKS] += 1

    return max(banks_unique_access_counts.values(), default=0)


def _get_b_tile_float4_read_indices(
    layout: KernelLayout,
    phase: int,
    warp_id: int,
    read_offset_x: int,
    subtile_x: int,
    dot_idx: int,
    coord_mapper: TileCoordMapping,
) -> list[int]:
    phase_start_thread_tile_id = phase * NUM_THREADS_PER_FLOAT4_PHASE
    warp_x = warp_id % layout.warp_tiles_per_block_n
    warp_start_x = warp_x * layout.warp_tile_n
    read_indices = []

    for thread_id_in_phase in range(NUM_THREADS_PER_FLOAT4_PHASE):
        warp_thread_id = phase_start_thread_tile_id + thread_id_in_phase
        thread_tile_x = warp_thread_id % layout.thread_tiles_per_warp_subtile_n
        logical_start_x = (
            warp_start_x +
            subtile_x * layout.warp_subtile_n +
            thread_tile_x * layout.cols_per_thread_tile +
            read_offset_x
        )

        for float_idx_in_float4 in range(FLOATS_PER_FLOAT4):
            logical_x = logical_start_x + float_idx_in_float4
            physical_x, physical_y = coord_mapper(logical_x, dot_idx)
            physical_idx = physical_y * layout.tile_n + physical_x
            read_indices.append(physical_idx)

    return read_indices


def _get_b_tile_float4_read_conflict_degree(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> int:
    num_phases = THREADS_PER_WARP // NUM_THREADS_PER_FLOAT4_PHASE
    num_warps = layout.num_threads // THREADS_PER_WARP
    largest_conflict_degree = 0

    for warp_id in range(num_warps):
        for phase in range(num_phases):
            for dot_idx in range(layout.tile_k):
                for subtile_x in range(layout.warp_subtiles_per_warp_tile_n):
                    for read_offset_x in range(0, layout.cols_per_thread_tile, FLOATS_PER_FLOAT4):

                        phase_indices = _get_b_tile_float4_read_indices(
                            layout=layout,
                            phase=phase,
                            warp_id=warp_id,
                            read_offset_x=read_offset_x,
                            subtile_x=subtile_x,
                            dot_idx=dot_idx,
                            coord_mapper=coord_mapper,
                        )
                        conflict_degree = _get_bank_conflict_degree(phase_indices)
                        largest_conflict_degree = max(largest_conflict_degree, conflict_degree)

    return largest_conflict_degree


def _get_b_tile_float4_store_indices(
    layout: KernelLayout,
    phase: int,
    warp_id: int,
    tile_row_offset: int,
    coord_mapper: TileCoordMapping,
) -> list[int]:
    phase_start_thread_tile_id = phase * NUM_THREADS_PER_FLOAT4_PHASE
    store_indices = []

    for thread_id_in_phase in range(NUM_THREADS_PER_FLOAT4_PHASE):
        warp_thread_id = phase_start_thread_tile_id + thread_id_in_phase
        tid = warp_id * THREADS_PER_WARP + warp_thread_id

        logical_start_idx = tid * FLOATS_PER_FLOAT4
        logical_y = tile_row_offset + logical_start_idx // layout.tile_n
        logical_start_x = logical_start_idx % layout.tile_n

        for float_idx_in_float4 in range(FLOATS_PER_FLOAT4):
            logical_x = logical_start_x + float_idx_in_float4
            physical_x, physical_y = coord_mapper(logical_x, logical_y)
            physical_idx = physical_y * layout.tile_n + physical_x
            store_indices.append(physical_idx)

    return store_indices


def _get_b_tile_float4_store_conflict_degree(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> int:
    num_phases = THREADS_PER_WARP // NUM_THREADS_PER_FLOAT4_PHASE
    num_warps = layout.num_threads // THREADS_PER_WARP
    largest_conflict_degree = 0

    for tile_row_offset in range(0, layout.tile_k, layout.b_tile_row_stride):
        for warp_id in range(num_warps):
            for phase in range(num_phases):

                phase_indices = _get_b_tile_float4_store_indices(
                    layout=layout,
                    phase=phase,
                    warp_id=warp_id,
                    tile_row_offset=tile_row_offset,
                    coord_mapper=coord_mapper,
                )
                conflict_degree = _get_bank_conflict_degree(phase_indices)
                largest_conflict_degree = max(largest_conflict_degree, conflict_degree)

    return largest_conflict_degree


def _is_b_tile_mapping_bijective(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> bool:
    seen_indices = set()

    for logical_y in range(layout.tile_k):
        for logical_x in range(layout.tile_n):

            physical_x, physical_y = coord_mapper(logical_x, logical_y)
            if physical_y < 0 or physical_y >= layout.tile_k:
                return False
            if physical_x < 0 or physical_x >= layout.tile_n:
                return False

            physical_idx = physical_y * layout.tile_n + physical_x
            if physical_idx in seen_indices:
                return False

            seen_indices.add(physical_idx)

    return True


def _perserves_b_tile_float4_access(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> bool:
    for logical_y in range(layout.tile_k):
        for logical_start_x in range(0, layout.tile_n, FLOATS_PER_FLOAT4):
            physical_x0, physical_y0 = coord_mapper(logical_start_x, logical_y)
            if physical_x0 % FLOATS_PER_FLOAT4 != 0:
                return False

            for i in range(1, FLOATS_PER_FLOAT4):
                physical_xi, physical_yi = coord_mapper(logical_start_x + i, logical_y)
                if physical_xi != physical_x0 + i or physical_yi != physical_y0:
                    return False

    return True


def _make_b_tile_coord_mapper(bank_section_bit_mask: int) -> TileCoordMapping:
    def _coord_mapper(logical_x: int, logical_y: int) -> tuple[int, int]:
        logical_float4_x = logical_x // FLOATS_PER_FLOAT4
        offset_in_float4 = logical_x % FLOATS_PER_FLOAT4

        logical_bank_section_idx = logical_float4_x // FLOAT4_GROUPS_PER_BANK_REPEAT
        float4_chunk_idx_xor_mask = logical_bank_section_idx & bank_section_bit_mask

        physical_float4_idx = logical_float4_x ^ float4_chunk_idx_xor_mask
        physical_x = physical_float4_idx * FLOATS_PER_FLOAT4 + offset_in_float4
        return physical_x, logical_y
    return _coord_mapper


@cache
def find_b_tile_mapping_mask(layout: KernelLayout) -> int | None:
    for bit_mask in range(FLOAT4_GROUPS_PER_BANK_REPEAT):
        coord_mapper = _make_b_tile_coord_mapper(bit_mask)
        if (
            not _is_b_tile_mapping_bijective(layout, coord_mapper) or
            not _perserves_b_tile_float4_access(layout, coord_mapper)
        ):
            continue

        read_conflicts_degree = _get_b_tile_float4_read_conflict_degree(layout, coord_mapper)
        if read_conflicts_degree > CONFLICT_FREE_DEGREE:
            continue

        store_conflicts_degree = _get_b_tile_float4_store_conflict_degree(layout, coord_mapper)
        if store_conflicts_degree > CONFLICT_FREE_DEGREE:
            continue

        return bit_mask

    return None


def _get_a_tile_scalar_store_indices(
    layout: KernelLayout,
    warp_id: int,
    tile_row_offset: int,
    float_idx_in_float4: int,
    coord_mapper: TileCoordMapping,
) -> list[int]:
    store_indices = []

    for warp_thread_id in range(THREADS_PER_WARP):
        tid = warp_id * THREADS_PER_WARP + warp_thread_id
        logical_start_idx = tid * FLOATS_PER_FLOAT4

        logical_y = tile_row_offset + logical_start_idx // layout.tile_k
        logical_x = logical_start_idx % layout.tile_k + float_idx_in_float4

        physical_x, physical_y = coord_mapper(logical_x, logical_y)
        physical_idx = physical_y * layout.tile_m + physical_x
        store_indices.append(physical_idx)

    return store_indices


def _get_a_tile_scalar_store_conflict_degree(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> int:
    num_warps = layout.num_threads // THREADS_PER_WARP
    largest_conflict_degree = 0

    for tile_row_offset in range(0, layout.tile_m, layout.a_tile_row_stride):
        for warp_id in range(num_warps):
            for float_idx_in_float4 in range(FLOATS_PER_FLOAT4):

                phase_indices = _get_a_tile_scalar_store_indices(
                    layout=layout,
                    warp_id=warp_id,
                    tile_row_offset=tile_row_offset,
                    float_idx_in_float4=float_idx_in_float4,
                    coord_mapper=coord_mapper,
                )
                conflict_degree = _get_bank_conflict_degree(phase_indices)
                largest_conflict_degree = max(conflict_degree, largest_conflict_degree)

    return largest_conflict_degree


def _get_a_tile_float4_read_indices(
    layout: KernelLayout,
    phase: int,
    warp_id: int,
    dot_idx: int,
    read_offset_y: int,
    subtile_y: int,
    coord_mapper: TileCoordMapping
) -> list[int]:
    phase_start_thread_id = phase * NUM_THREADS_PER_FLOAT4_PHASE
    warp_y = warp_id // layout.warp_tiles_per_block_n
    warp_start_y = warp_y * layout.warp_tile_m
    read_indices = []

    for thread_id_in_phase in range(NUM_THREADS_PER_FLOAT4_PHASE):
        warp_thread_id = phase_start_thread_id + thread_id_in_phase
        thread_tile_y = warp_thread_id // layout.thread_tiles_per_warp_subtile_n

        logical_start_y = (
            warp_start_y +
            subtile_y * layout.warp_subtile_m +
            thread_tile_y * layout.rows_per_thread_tile +
            read_offset_y
        )

        for float_idx_int_float4 in range(FLOATS_PER_FLOAT4):
            logical_y = logical_start_y + float_idx_int_float4
            physical_x, physical_y = coord_mapper(dot_idx, logical_y)
            physical_idx = physical_y * layout.tile_m + physical_x
            read_indices.append(physical_idx)

    return read_indices


def _get_a_tile_float4_read_conflict_degree(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> int:
    num_warps = layout.num_threads // THREADS_PER_WARP
    num_phases = THREADS_PER_WARP // NUM_THREADS_PER_FLOAT4_PHASE
    largest_conflict_degree = 0

    for warp_id in range(num_warps):
        for phase in range(num_phases):
            for dot_idx in range(layout.tile_k):
                for subtile_y in range(layout.warp_subtiles_per_warp_tile_m):
                    for read_offset_y in range(0, layout.rows_per_thread_tile, FLOATS_PER_FLOAT4):

                        phase_indices = _get_a_tile_float4_read_indices(
                            layout=layout,
                            phase=phase,
                            warp_id=warp_id,
                            dot_idx=dot_idx,
                            read_offset_y=read_offset_y,
                            subtile_y=subtile_y,
                            coord_mapper=coord_mapper,
                        )
                        conflict_degree = _get_bank_conflict_degree(phase_indices)
                        largest_conflict_degree = max(largest_conflict_degree, conflict_degree)

    return largest_conflict_degree


def _is_a_tile_mapping_bijective(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> bool:
    seen_indices = set()

    for logical_y in range(layout.tile_m):
        for logical_x in range(layout.tile_k):

            physical_x, physical_y = coord_mapper(logical_x, logical_y)
            if physical_y < 0 or physical_y >= layout.tile_k:
                return False
            if physical_x < 0 or physical_x >= layout.tile_m:
                return False

            physical_idx = physical_y * layout.tile_m + physical_x
            if physical_idx in seen_indices:
                return False

            seen_indices.add(physical_idx)

    return True


def _perserves_a_tile_scalar_store_pattern(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> bool:
    for logical_y in range(layout.tile_m):
        for logical_start_x in range(0, layout.tile_k, FLOATS_PER_FLOAT4):

            physical_x0, physical_y0 = coord_mapper(logical_start_x, logical_y)
            for i in range(1, FLOATS_PER_FLOAT4):
                physical_xi, physical_yi = coord_mapper(logical_start_x + i, logical_y)
                if physical_xi != physical_x0 or physical_yi != physical_y0 + i:
                    return False

    return True


def _preserves_a_tile_float4_reads(
    layout: KernelLayout,
    coord_mapper: TileCoordMapping,
) -> bool:
    for logical_x in range(layout.tile_k):
        for logical_start_y in range(0, layout.tile_m, FLOATS_PER_FLOAT4):
            physical_x0, physical_y0 = coord_mapper(logical_x, logical_start_y)
            if physical_x0 % FLOATS_PER_FLOAT4 != 0:
                return False

            for i in range(1, FLOATS_PER_FLOAT4):
                physical_xi, physical_yi = coord_mapper(logical_x, logical_start_y + i)
                if physical_xi != physical_x0 + i or physical_yi != physical_y0:
                    return False

    return True


def _make_a_tile_coord_mapper(
    layout: KernelLayout,
    bank_section_bit_mask: int,
    k_group_shift: int,
) -> TileCoordMapping:
    float4_bank_position_mask = min(
        FLOAT4_GROUPS_PER_BANK_REPEAT - 1,
        layout.tile_m // FLOATS_PER_FLOAT4 - 1,
    )

    def _coord_mapper(logical_x: int, logical_y: int) -> tuple[int, int]:
        logical_float4_y = logical_y // FLOATS_PER_FLOAT4
        offset_in_float4 = logical_y % FLOATS_PER_FLOAT4

        bank_section_idx = logical_float4_y // FLOAT4_GROUPS_PER_BANK_REPEAT
        bank_section_xor_mask = bank_section_idx & bank_section_bit_mask

        logical_k_group_idx = logical_x // FLOATS_PER_FLOAT4
        k_group_xor_mask = (logical_k_group_idx << k_group_shift) & float4_bank_position_mask

        physical_float4_idx = logical_float4_y ^ bank_section_xor_mask ^ k_group_xor_mask
        physical_x = physical_float4_idx * FLOATS_PER_FLOAT4 + offset_in_float4
        return physical_x, logical_x

    return _coord_mapper


@cache
def find_a_tile_mapping_mask_and_shift(layout: KernelLayout) -> tuple[int, int] | None:
    for bank_section_bit_mask in range(FLOAT4_GROUPS_PER_BANK_REPEAT):
        for k_group_shift in A_K_GROUP_SHIFT_CANDIDATES:

            coord_mapper = _make_a_tile_coord_mapper(layout, bank_section_bit_mask, k_group_shift)
            if (
                not _is_a_tile_mapping_bijective(layout, coord_mapper) or
                not _perserves_a_tile_scalar_store_pattern(layout, coord_mapper) or
                not _preserves_a_tile_float4_reads(layout, coord_mapper)
            ):
                continue

            read_conflicts = _get_a_tile_float4_read_conflict_degree(layout, coord_mapper)
            if read_conflicts > CONFLICT_FREE_DEGREE:
                continue

            store_conflicts = _get_a_tile_scalar_store_conflict_degree(layout, coord_mapper)
            if store_conflicts > CONFLICT_FREE_DEGREE:
                continue

            return bank_section_bit_mask, k_group_shift

    return None
