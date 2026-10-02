from pathlib import Path

THREADS_PER_WARP = 32
FLOATS_PER_FLOAT4 = 4
CUDA_FILE_EXT = "cu"
OBJECT_FILE_EXT = "o"

ROOT_DIR = Path(__file__).parent.parent
GENERATED_DIR = ROOT_DIR / "build" / "generated"
KERNELS_DIR = GENERATED_DIR / "kernels"
OBJECTS_DIR = GENERATED_DIR / "objects"

KERNEL_TEMPLATE_PATH = ROOT_DIR / "kernel.cu.in"
