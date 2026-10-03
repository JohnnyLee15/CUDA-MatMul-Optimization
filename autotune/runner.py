import subprocess
from pathlib import Path

from .constants import KERNEL_RESOURCE_LIMIT_EXIT_CODE


def run_benchmark(
    executable_path: Path,
    m: int,
    n: int,
    k: int,
) -> float | None:
    command = [str(executable_path), str(m), str(n), str(k),]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode == KERNEL_RESOURCE_LIMIT_EXIT_CODE:
            return None
        raise

    return float(result.stdout.strip())
