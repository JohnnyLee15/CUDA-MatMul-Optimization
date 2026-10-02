from .tuner import run_sweep
from .results import write_results
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
    results_path = write_results(results, MATRIX_SIZES)
    print(f"Results saved to {results_path} | Saved top {KEEP_TOP_K_KERNELS} kernels")
