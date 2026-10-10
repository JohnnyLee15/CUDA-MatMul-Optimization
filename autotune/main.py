from .tuner import run_sweep
from .config import MatrixSizes


KEEP_TOP_K_KERNELS = 5
MATRIX_SIZES = MatrixSizes(
    m = 4096,
    n = 4096,
    k = 4096,
)


if __name__ == "__main__":
    print(f"Running Sweep for M={MATRIX_SIZES.m}, N={MATRIX_SIZES.n}, K={MATRIX_SIZES.k}")
    results = run_sweep(MATRIX_SIZES, KEEP_TOP_K_KERNELS)
    print(f"Sweep complete. Kept {min(KEEP_TOP_K_KERNELS, len(results))} kernels.")
