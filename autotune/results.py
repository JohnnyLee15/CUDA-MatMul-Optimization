from pathlib import Path

from .config import MatrixSizes
from .constants import RESULT_FILE_PREFIX, RESULTS_DIR


RANK_COL_NAME = "Rank"
CONFIG_COL_NAME = "Config"
AVG_DURATION_COL_NAME = "Avg_Duration_Ms"
GAP = " | "
HEADER_SEPARATOR_CHAR = "-"


def _get_col_widths(benchmark_results: list[tuple[str, float, Path]]) -> tuple[int, int, int]:
    width_rank = max(len(RANK_COL_NAME), len(str(len(benchmark_results))))
    width_config = len(CONFIG_COL_NAME)
    width_duration = len(AVG_DURATION_COL_NAME)

    for name, duration, _ in benchmark_results:
        width_config = max(width_config, len(name))
        width_duration = max(width_duration, len(str(duration)))

    return width_rank, width_config, width_duration


def write_results(benchmark_results: list[tuple[str, float, Path]], matrix_sizes: MatrixSizes) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_filename = (
        f"{RESULT_FILE_PREFIX}_"
        f"m={matrix_sizes.m}_"
        f"n={matrix_sizes.n}_"
        f"k={matrix_sizes.k}.txt"
    )

    results_path = RESULTS_DIR / results_filename
    width_rank, width_config, width_duration = _get_col_widths(benchmark_results)

    with open(results_path, "w", encoding="utf-8") as file:
        file.write(
            f"{RANK_COL_NAME:<{width_rank}}{GAP}"
            f"{CONFIG_COL_NAME:<{width_config}}{GAP}"
            f"{AVG_DURATION_COL_NAME:<{width_duration}}\n"
        )

        file.write(
            f"{HEADER_SEPARATOR_CHAR * width_rank}{GAP}"
            f"{HEADER_SEPARATOR_CHAR * width_config}{GAP}"
            f"{HEADER_SEPARATOR_CHAR * width_duration}\n"
        )

        for rank, (name, duration, _) in enumerate(benchmark_results, start=1):
            file.write(
                f"{rank:<{width_rank}}{GAP}"
                f"{name:<{width_config}}{GAP}"
                f"{duration:>{width_duration}}\n"
            )

    return results_path
