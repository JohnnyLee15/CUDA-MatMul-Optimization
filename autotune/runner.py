import subprocess
from pathlib import Path


def run_benchmark(
    executable_path: Path,
    m: int,
    n: int,
    k: int,
) -> float:
    command = [str(executable_path), str(m), str(n), str(k),]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())
