from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
RESULTS_DIR = BENCHMARKS_DIR / "results"
PLOTS_DIR = BENCHMARKS_DIR / "plots"


def read_benchmark(path: Path) -> tuple[int, dict[str, float]]:
    dimensions = path.stem.rsplit("_", 1)[1]
    m, n, k = map(int, dimensions.split("x"))

    if not m == n == k:
        raise ValueError(f"Expected square matrices in {path.name}")

    kernel_name = None
    kernel_to_gflops = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith("Benchmarking "):
            kernel_name = line.removeprefix("Benchmarking ")

        elif line.startswith("Performance: ") and kernel_name is not None:
            gflops = float(line.split()[1])
            kernel_to_gflops[kernel_name] = gflops

    return m, kernel_to_gflops


def main() -> None:
    results = {}

    for path in sorted(RESULTS_DIR.glob("benchmark_rtx3080_*.txt")):
        matrix_size, kernel_to_gflops = read_benchmark(path)

        for kernel_name, gflops in kernel_to_gflops.items():
            results.setdefault(kernel_name, {})[matrix_size] = gflops

    if not results:
        raise ValueError(f"No benchmark results found in {RESULTS_DIR}")

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap("tab20")

    for i, (kernel_name, measurements) in enumerate(results.items()):
        sizes = sorted(measurements)
        gflops = [measurements[size] for size in sizes]
        label = kernel_name.removeprefix("GPU ")

        if kernel_name == "GPU Autotune Exact Fit":
            label += " (tuned for 4096)"

        is_cublas = kernel_name == "cuBLAS SGEMM"

        ax.plot(
            sizes,
            gflops,
            label=label,
            marker="o",
            color="black" if is_cublas else colors(i),
            linestyle="--" if is_cublas else "-",
            linewidth=2,
        )

    all_sizes = sorted({
        size
        for measurements in results.values()
        for size in measurements
    })

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(size) for size in all_sizes])
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Square matrix dimension (M = N = K)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title("CUDA Matrix Multiplication Performance on RTX 3080")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "matmul_throughput_rtx3080.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()
