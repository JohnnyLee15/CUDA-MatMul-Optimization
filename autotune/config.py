from dataclasses import dataclass

from .constants import FLOATS_PER_FLOAT4


@dataclass(frozen=True)
class MatrixSizes:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class KernelLayout:
    block_x: int
    block_y: int

    rows_per_thread_tile: int
    cols_per_thread_tile: int

    tile_k: int

    warp_tile_m: int
    warp_tile_n: int

    warp_subtiles_per_warp_tile_m: int
    warp_subtiles_per_warp_tile_n: int


    @property
    def num_threads(self) -> int:
        return self.block_x * self.block_y


    @property
    def tile_m(self) -> int:
        return self.block_y * self.warp_subtiles_per_warp_tile_m * self.rows_per_thread_tile


    @property
    def tile_n(self) -> int:
        return self.block_x * self.warp_subtiles_per_warp_tile_n * self.cols_per_thread_tile


    @property
    def warp_subtile_m(self) -> int:
        return self.warp_tile_m // self.warp_subtiles_per_warp_tile_m

    @property
    def warp_subtile_n(self) -> int:
        return self.warp_tile_n // self.warp_subtiles_per_warp_tile_n


    @property
    def warp_tiles_per_block_n(self) -> int:
        return self.tile_n // self.warp_tile_n


    @property
    def thread_tiles_per_warp_subtile_n(self) -> int:
        return self.warp_subtile_n // self.cols_per_thread_tile


    @property
    def floats_per_tile_load_pass(self) -> int:
        return self.num_threads * FLOATS_PER_FLOAT4


    @property
    def a_tile_row_stride(self) -> int:
        return self.floats_per_tile_load_pass // self.tile_k


    @property
    def b_tile_row_stride(self) -> int:
        return self.floats_per_tile_load_pass // self.tile_n


@dataclass(frozen=True)
class Config:
    layout: KernelLayout
    acc_loop_unroll: int


    @property
    def name(self) -> str:
        return (
            f"bx_{self.layout.block_x}_"
            f"by_{self.layout.block_y}_"
            f"wsm_{self.layout.warp_subtiles_per_warp_tile_m}_"
            f"wsn_{self.layout.warp_subtiles_per_warp_tile_n}_"
            f"ttm_{self.layout.rows_per_thread_tile}_"
            f"ttn_{self.layout.cols_per_thread_tile}_"
            f"btk_{self.layout.tile_k}_"
            f"wtm_{self.layout.warp_tile_m}_"
            f"wtn_{self.layout.warp_tile_n}_"
            f"unroll_{self.acc_loop_unroll}"
        )


@dataclass(frozen=True)
class ATileMappingParams:
    bank_section_bit_mask: int
    k_group_bank_bit_shift: int


@dataclass(frozen=True)
class BTileMappingParams:
    bank_section_bit_mask: int


@dataclass(frozen=True)
class KernelSpec:
    config: Config
    a_tile_mapping: ATileMappingParams
    b_tile_mapping: BTileMappingParams

