from dataclasses import dataclass

from .constants import FLOATS_PER_FLOAT4


@dataclass(frozen=True)
class Config:
    block_x: int
    block_y: int

    rows_per_tile_thread: int
    cols_per_tile_thread: int

    tile_k: int

    warp_tile_m: int
    warp_tile_n: int

    acc_loop_unroll: int


    @property
    def num_threads(self) -> int:
        return self.block_x * self.block_y


    @property
    def tile_m(self) -> int:
        return self.block_y * self.rows_per_tile_thread


    @property
    def tile_n(self) -> int:
        return self.block_x * self.cols_per_tile_thread


    @property
    def warp_tiles_per_block_n(self) -> int:
        return self.tile_n // self.warp_tile_n


    @property
    def thread_tiles_per_warp_n(self) -> int:
        return self.warp_tile_n // self.cols_per_tile_thread


    @property
    def floats_per_tile_load_pass(self) -> int:
        return self.num_threads * FLOATS_PER_FLOAT4


    @property
    def a_tile_row_stride(self) -> int:
        return self.floats_per_tile_load_pass // self.tile_k


    @property
    def b_tile_row_stride(self) -> int:
        return self.floats_per_tile_load_pass // self.tile_n


    @property
    def name(self) -> str:
        return (
            f"bx_{self.block_x}_"
            f"by_{self.block_y}_"
            f"ttm_{self.rows_per_tile_thread}_"
            f"ttn_{self.cols_per_tile_thread}_"
            f"btk_{self.tile_k}_"
            f"wtm_{self.warp_tile_m}_"
            f"wtn_{self.warp_tile_n}_"
            f"unroll{self.acc_loop_unroll}"
        )
