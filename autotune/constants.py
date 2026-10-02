from pathlib import Path

THREADS_PER_WARP = 32
FLOATS_PER_FLOAT4 = 4

CUDA_FILE_EXT = "cu"
OBJECT_FILE_EXT = "o"
RESULT_FILE_PREFIX = "result"

ROOT_DIR = Path(__file__).parent.parent
GENERATED_DIR = ROOT_DIR / "build" / "generated"
EXECUTABLES_DIR = GENERATED_DIR / "executables"
KERNELS_DIR = GENERATED_DIR / "kernels"
OBJECTS_DIR = GENERATED_DIR / "objects"
RESULTS_DIR = GENERATED_DIR / "results"

KERNEL_TEMPLATE_PATH = ROOT_DIR / "autotune" / "kernel.cu.in"
