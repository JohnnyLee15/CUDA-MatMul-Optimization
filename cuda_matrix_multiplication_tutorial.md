# CUDA Matrix Optimization

This tutorial follows the process of optimizing matrix multiplication on a CUDA GPU. We start with a straightforward kernel and improve it one step at a time, focusing on how threads access memory, reuse data, and divide up the calculation.

We begin with CUDA indexing and warps, then move through coalescing, shared memory, register tiling, vectorization, and shared memory bank conflicts. Each section builds on the earlier kernels and explains the reason for the next change.

## Contents

- [Calculating Indexes](#calculating-indexes)
- [Threads, Blocks, and Warps](#threads-blocks-and-warps)
- [Optimization 1: Global Memory Coalescing](#matrix-kernel-optimization-1-global-memory-coalescing)
- [Optimization 2: Shared Memory](#matrix-kernel-optimization-2-shared-memory)
- [Optimization 3: 1D Register Tiling](#matrix-kernel-optimization-3-1d-register-tiling)
- [Optimization 4: 2D Register Tiling](#matrix-kernel-optimization-4-2d-register-tiling)
- [Optimization 5: Vectorization](#matrix-kernel-optimization-5-vectorization)
- [Optimization 6: Resolving Bank Conflicts](#matrix-kernel-optimization-6-resolving-bank-conflicts)
- [Optimization 7: Warp Tiling](#matrix-kernel-optimization-7-warp-tiling)

## Calculating Indexes

CUDA uses `x`, `y`, and `z` coordinates to identify blocks (`blockIdx`, block = group of threads) and threads within each block (`threadIdx`). Dimensions that aren’t used have size 1.

When numbering threads within a block, `x` varies fastest, then `y`, then `z`. Blocks are also numbered this way.

`gridDim` gives the dimensions of the grid of blocks. `blockDim` gives the dimensions of each block of threads.

Coordinate numbering does not determine execution order. Within a block, it determines which threads belong to each warp. Blocks and warps are not guaranteed to execute in numerical order.

In the following examples, `x` is the column, `y` is the row, and `z` is the plane. Elements are stored consecutively.

### 2D Coordinates Example

Say we have an $M \times N$ matrix $A$ that is row major. That is $A = [a_{1, 1}, a_{1, 2}, \dots a_{1, N}, a_{2, 1}, a_{2, 2}, \dots a_{M, N}]$

The storage order illustration uses one based subscripts. The indexing formulas below use zero based coordinates.

First, we go from coordinates to a flat index. Given 0 indexed coordinates `(x, y)`:

```text
i = y * N + x
```

Each row contains $N$ elements. Since `y` tells us how many rows come before the current row, `y * N` skips those rows and puts us at the start of the current row. Adding `x` moves us to the desired element within that row.

Now, we go from a flat index to coordinates. Given index `i`:

```text
x = i % N
y = i / N
```

The index `i` tells us how many elements come before the current element.

Each row contains `N` elements. The remainder, `i % N`, tells us how many elements are left after removing all complete rows. That is how many positions into the current row we are, giving the zero based `x` coordinate.

Using integer division, `i / N` counts how many complete rows the `i` preceding elements fill. That is the zero based `y` coordinate: row 0 has no rows before it, row 1 has one, and so on.

### 3D Coordinates Example

Say we have a $Z \times Y \times X$ tensor $A$ that is row major. That is $A = [a_{1, 1, 1}, a_{1, 1, 2}, \dots a_{1, 1, X}, a_{1, 2, 1}, a_{1, 2, 2}, \dots a_{1, Y, X}, a_{2, 1, 1}, \dots, a_{Z, Y, X}]$

The storage order illustration uses one based subscripts. The indexing formulas below use zero based coordinates.

First, we go from coordinates to a flat index. Given 0 indexed coordinates `(x, y, z)`:

```text
i = (z * (X * Y)) + (y * X) + x = (z * Y + y) * X + x
```

Each plane contains `X * Y` elements. Multiplying that by `z` skips the planes before the current plane. Then `y * X` skips the rows before the current row within our current plane. Finally, adding `x` moves us to the desired element within that row.

Now, we go from flat index to coordinates. Given `i`:

```text
x = i % X
y = (i / X) % Y
z = i / (X * Y)
```

Each row contains `X` elements. The remainder, `i % X`, tells us how many elements are left after removing all complete rows. That is how many positions into the current row we are, giving the zero based `x` coordinate.

Using integer division, `i / X` counts how many complete rows the `i` preceding elements fill, including rows in earlier planes. Each plane contains `Y` rows, so taking `(i / X) % Y` removes the rows in all complete planes. What remains is the number of rows before the current row within the current plane. That is the zero based `y` coordinate: row 0 has no rows before it within the plane, row 1 has one, and so on.

Each plane contains `X * Y` elements. Using integer division, `i / (X * Y)` counts how many complete planes the `i` preceding elements fill. That is the zero based `z` coordinate: plane 0 has no planes before it, plane 1 has one, and so on.

We can write the flat index as `i = (z * Y + y) * X + x`. Here, `z * Y` counts all rows in the planes before the current plane. Adding `y` includes the rows before the current row within this plane. So `z * Y + y` is the total number of rows before the current row. Each row contains `X` elements, so multiplying by `X` gives the number of elements in those rows. Finally, adding `x` moves us to the desired position within the current row.

> **Note:** Starting with the slowest varying coordinate `z`, combine it with `y`: `w = z * Y + y`. Then combine `w` with `x`: `i= w * X + x`. This generalizes to any number of dimensions - multiply by the next dimension’s size, then add its coordinate.

## Threads, Blocks, and Warps

When launching a CUDA kernel, we specify the number of blocks in the grid and the number of threads in each block. The grid contains all threads we want to launch. As mentioned earlier, both can be arranged in up to three dimensions `x`, `y`, and `z`. Threads within a block and blocks within a grid are numbered with `x` varying the fastest, followed by `y`, then `z`.

Threads within a block are grouped consecutively into warps, each with 32 thread positions. A warp belongs to one block and never spans multiple blocks. If the block’s thread count is not a multiple of 32, the last warp has unused positions that cannot be filled with threads from another block, leaving some of that warp’s execution capacity unused. For example, suppose we process 640 elements with one thread per element:

- 20 threads per block: 32 blocks, each containing one partially filled warp → 32 warps total
- 32 threads per block: 20 blocks, each containing one full warp → 20 warps total

Both launch 640 threads, but the first arrangement spreads the same useful work across more warps. For the same per thread instruction sequence, this means more warp instructions must be issued. It can also hit the SM’s resident block limit sooner. Thus, we should in most cases choose a block size that is a multiple of 32.

Warps matter for memory performance because requests from threads executing the same global memory instruction can be served by the same transaction when they fall within the same memory segment. We’ll explore this in the coalescing section.

For example, a `16 x 16` block contains 256 threads, forming 8 `(256 / 32 = 8)` warps. Its first warp contains:

```text
Threads  0–15: (0, 0), (1, 0), ... (15, 0)
Threads 16–31: (0, 1), (1, 1), ... (15, 1)
```

When a CUDA kernel is launched, the GPU assigns each block to an SM (streaming multiprocessor), which contains hardware for executing instructions, including arithmetic and memory operations. An SM can hold multiple blocks and warps at once, subject to hardware and resource limits. These are called resident blocks and warps.

Occupancy measures the number of resident warps as a fraction of the maximum number of warps the SM can hold. Each SM has both a resident block limit and a resident warp limit, and both apply at the same time. Small blocks can hit the resident block limit before filling the available warp capacity. Large blocks can also leave warp capacity unused if the number of warps within a block is larger than the number of warps currently available in the SM. Register and shared memory requirements can further limit how many blocks fit.

For example, suppose an SM can hold at most 48 resident warps and 16 resident blocks. Each warp contains 32 threads, so full occupancy corresponds to `48 * 32 = 1536` resident threads.

Ignoring register and shared memory limits, we need a block size that fills those 48 warp slots without exceeding 16 blocks. For block sizes that are multiples of 32, the choices that fit exactly are:

| Threads per block | Warps per block | Resident blocks | Resident warps |
| :---: | :---: | :---: | :---: |
| 96 | 3 | 16 | 48 |
| 128 | 4 | 12 | 48 |
| 192 | 6 | 8 | 48 |
| 256 | 8 | 6 | 48 |
| 384 | 12 | 4 | 48 |
| 512 | 16 | 3 | 48 |
| 768 | 24 | 2 | 48 |

These choices also need to satisfy the GPU’s per block thread limit.

With 64 threads per block, filling all 48 warps would require `1536 / 64 = 24` blocks. However, only 16 blocks can reside on the SM, giving `(64 * 16) / 32 = 32` resident warps or `32 / 48 = 2 / 3 = 66.6..%` occupancy.

With 544 threads per block, filling all 1536 thread positions would require `1536 / 544 = 2.82..` blocks. However, `0.82..` of a block cannot reside on the SM, only whole blocks can, thus, only 2 blocks reside on the SM, resulting in `(544 * 2) / 32 = 34` resident warps or about `71%` occupancy.

Maximum occupancy is not always the fastest choice. This example only shows how block size interacts with the block and warp limits.

The SM makes progress on all resident warps concurrently by interleaving their instructions. Warp schedulers select ready warps and issue their next instructions as the required execution resources become available. A warp is ready when its next instruction is no longer waiting on dependencies, such as the results of an earlier memory load, calculation, etc. This lets the SM execute work from other warps while some are waiting, helping hide memory latency as the SM spends less time idle.

This gives us the background needed to understand memory coalescing.

## Matrix Kernel Optimization 1: Global Memory Coalescing

### Coalescing

Memory coalescing refers to the GPU combining memory requests from threads in a warp into as few global memory transactions as possible. Having neighbouring threads access adjacent memory addresses / locations generally makes those transactions more efficient, because more of each fetched memory segment is used by the warp.

Precisely speaking, coalescing happens for the active threads in **one** warp, executing **one** memory instruction, where the GPU groups memory requests from threads within the warp into as few memory transactions as possible. Requests from different warps are not combined into the same memory instruction and are therefore coalesced separately, even if adjacent threads from different neighbouring warps access adjacent global memory. Separate memory instructions, such as loading from `ptr *a` and `ptr *b`, are analyzed separately, to see if the warp's memory access is coalesced.

A memory instruction happens at the warp level and tells threads in a warp to load or store data. A memory transaction is a logical memory system operation used to serve an instruction given to a warp. One instruction can require multiple transactions, depending on the addresses requested by the threads. Thus, the sequence is.

1. Warp executes a memory instruction
2. Each active thread/lane computes the address it needs
3. Those per thread memory requests are collected
4. The memory subsystem examines the requested addresses.
5. It generates the required memory transactions
6. Those transactions serve the warp’s requests

For newer NVIDIA GPUs, a warp's global memory accesses are served using 32 byte aligned memory segments - aligned meaning each segment begins at an address that is a multiple of 32 bytes. For example:

```text
Segment 0: Bytes  0-31
Segment 1: Bytes 32-63
Segment 2: Bytes 64-95
...
```

For a load where each thread requests one 4 byte float, the warp needs a transaction for each segment containing a requested value. Multiple threads requesting values within the same segment can have those requests served together. This also includes threads requesting the same value.

Let's assume we have an array of floats, where each float is 4 bytes. Now assume that the array's address is 32 byte aligned. Then,

```text
Segment 0:    a[0] through a[7]
Segment 1:   a[8] through a[15]
Segment 2:  a[16] through a[23]
...
```

Therefore:

- Threads reading `a[0]` and `a[7]` load one segment
- Threads reading `a[7]` and `a[8]` load two segments even though `a[7]` and `a[8]` are adjacent

> **Note:** the above example assumes threads are given the same instruction (for example, reading `a[i]`).

Now let's go through a precise example.

Say we have an array `a` of 8 floats that starts at a 32 byte aligned address. We launch one block containing 8 threads, so the block is scheduled as one warp with 8 active threads and 24 unused thread positions. Assume we have a kernel that looks like.

```cuda
__global__ void read(const float* __restrict__ a, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    a[i];
}
```

The expression `a[i]` corresponds to a **single** memory load instruction issued for the warp. Each of the 8 active threads executes that same instruction and generates one 4 byte memory request:

```text
thread 0 → a[0]
thread 1 → a[1]
...
thread 7 → a[7]
```

Since `a[0]` through `a[7]` occupy exactly 32 contiguous bytes and `a[0]` begins on a 32 byte aligned address, all 8 requests fit within **one** 32 byte memory segment. Therefore, all 8 memory requests are served by **one** memory transaction. Thus, we have perfect coalescing.

Overall, memory coalescing reduces the number of memory transactions required to serve a warp's memory instruction and can therefore improve kernel performance compared with a poorly coalesced access pattern.

### Naive Matrix Multiplication Kernel Review

Let's look at the Naive CUDA kernel and see if the access patterns coalesce efficiently.

```cuda
__global__ void matMulCudaNaiveKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= m || x >= n) return;

    float acc = 0.0f;
    for (int i = 0; i < k; i++) {
        acc += a[y * k + i] * b[i * n + x];
    }

    c[y * n + x] = acc;
}
```

The input matrices are `a` and `b`, and the output matrix is `c`. Their dimensions are:

- `a`: `m` rows and `k` columns
- `b`: `k` rows and `n` columns
- `c`: `m` rows and `n` columns

Assume `k = 256`, `n = 1024`, with `16 x 16` = `256` threads per block. Also assume the arrays begin at addresses aligned to 32 bytes and all threads in the warp we examine are within the matrix bounds.

Each thread is assigned to calculate one element of `c` specifically at index `c[y * n + x] == c[y][x]`.

We will examine the first warp of block `(0, 0)` at loop iteration `i = 0`. Denote a thread as $t_i$, $i \in \{0, 1, \dots, 31\}$. Their (`threadIdx.x`, `threadIdx.y`) coordinates are:

- $t_0: (0, 0), t_1: (1, 0), \dots, t_{15}: (15, 0)$
- $t_{16}: (0, 1), t_{17}: (1, 1), \dots, t_{31}: (15, 1)$

First, consider the load from `a`: `a[y * k + i]`

At this iteration, threads 0–15 have `y = 0`, so they all read `a[0]`. Threads 16–31 have `y = 1`, so they all read `a[256]`. These two values fall in different 32 byte segments:

```text
Segment  0:     a[0] through a[7]  - threads 0-15 request a[0], but we fetch all 8 floats in the segment
Segment 32: a[256] through a[263]  - threads 16-31 request a[256], but we fetch all 8 floats in the segment
```

The hardware combines the repeated requests from threads 0–15 into one memory transaction and those from threads 16–31 into another. Thus, two memory transactions are needed to load the values requested by the warp.

However, the threads only use one distinct 4 byte float from each 32 byte segment in this load. The remaining floats are not requested at this iteration, although later iterations may benefit from them being cached.

So this access pattern combines many requests into few transactions, but uses only a small part of each segment. These are two different aspects of memory-access efficiency.

Next, consider the load from `b`: `b[i * n + x]`

| Threads      | `x`  | Element read |
|--------------|----|--------------|
| `t_0, t_16`  | 0  | `b[0]`       |
| `t_1, t_17`  | 1  | `b[1]`       |
| `t_2, t_18`  | 2  | `b[2]`       |
| ⋮             | ⋮  | ⋮            |
| `t_15, t_31` | 15 | `b[15]`      |

The 32 threads read 16 different contiguous values from `b`, indices 0-15. These occupy two consecutive 32 byte segments:

```text
Segment 0:  b[0] through b[7]  - threads 0-7, and 16-23
Segment 1: b[8] through b[15]  - threads 8-15, and 24-31
```

The hardware combines the requests from threads 0-7, and 16-23 into one memory transaction and those from threads 8-15, and 24-31 into another. Thus, two memory transactions are needed to load the values requested by the warp.

Here, both memory transactions are fully utilized because every float in each segment is used by the warp. The 16 distinct floats occupy 64 bytes, so two 32 byte transactions are the minimum needed to load them.

Here is an example of perfect coalescing.

Finally, after the loop finishes, consider the store to `c`: `c[y * n + x]`.

| Thread | y | x  | Element written |
|--------|---|----|-----------------|
| `t_0`  | 0 | 0  | `c[0]`          |
| `t_1`  | 0 | 1  | `c[1]`          |
| `t_2`  | 0 | 2  | `c[2]`          |
| ⋮      | ⋮ | ⋮  | ⋮               |
| `t_15` | 0 | 15 | `c[15]`         |
| `t_16` | 1 | 0  | `c[1024]`       |
| `t_17` | 1 | 1  | `c[1025]`       |
| `t_18` | 1 | 2  | `c[1026]`       |
| ⋮      | ⋮ | ⋮  | ⋮               |
| `t_31` | 1 | 15 | `c[1039]`       |

The 32 threads write to 32 different addresses of `c`. These occupy four 32 byte segments:

```text
Segment 0:         c[0] through c[7]  - threads 0-7
Segment 1:        c[8] through c[15]  - threads 8-15
Segment 128: c[1024] through c[1031]  - threads 16-23
Segment 129: c[1032] through c[1039]  - threads 24-31
```

The hardware combines the write requests from threads 0-7 into one memory transaction, threads 8-15 into another, threads 16-23 into another, and threads 24-31 into another. Thus, four memory transactions are needed to write the warp's results.

Here, all four memory transactions are fully utilized because the warp writes to every float in each segment. The 32 distinct floats occupy 128 bytes, so four 32 byte transactions are the minimum needed to write them.

Here is another example of perfect coalescing.

Our naive kernel already coalesces its global memory accesses. `b` reads and `c` writes fully utilize the accessed segments under the conditions analyzed above, while `a` reads combine repeated requests but use only a small portion of each segment per iteration.

Next, we’ll explore shared memory tiling, where threads in a block cooperate to load tiles of `a` and `b` and reuse those values across multiple calculations, reducing repeated global memory loads.

## Matrix Kernel Optimization 2: Shared Memory

Shared memory is fast, programmer managed memory located on an SM. It is commonly used by first loading heavily reused data from slower global memory (VRAM) into shared memory, then repeatedly accessing that data from shared memory throughout the kernel.

Instead of repeatedly accessing the same data from global memory, load it into shared memory once and reuse it from there.

Shared memory is shared among the threads within a single thread block. Therefore, when optimizing the use of shared memory, we mainly care about how the threads within a block cooperatively load data into shared memory and how they reuse that data during computation.

Below is the shared memory kernel:

```cuda
constexpr uint32_t TILE_K = 16;

__global__ void matMulCudaSharedMemKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[BLOCK_Y][TILE_K];
    __shared__ float bTile[TILE_K][BLOCK_X];

    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    uint32_t cy = blockIdx.y * blockDim.y + ty;
    uint32_t cx = blockIdx.x * blockDim.x + tx;

    float acc = 0.0f;
    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {
        if (tx < TILE_K) {
            uint32_t ax = tileStart + tx;
            aTile[ty][tx] = (cy < m && ax < k) ? a[cy * k + ax] : 0.0f;
        }

        if (ty < TILE_K) {
            uint32_t by = tileStart + ty;
            bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
        }

        __syncthreads();

        for (uint32_t i = 0; i < TILE_K; i++) {
            acc += aTile[ty][i] * bTile[i][tx];
        }

        __syncthreads();
    }

    if (cy < m && cx < n) {
        c[cy * n + cx] = acc;
    }
}
```

The input matrices are `a` and `b`, and the output matrix is `c`. Their dimensions are:

- `a`: `m` rows and `k` columns
- `b`: `k` rows and `n` columns
- `c`: `m` rows and `n` columns

Now, assume `BLOCK_X = BLOCK_Y = TILE_K = 16`. Thus, each thread block contains `BLOCK_Y * BLOCK_X = 256` threads.

Each thread is assigned to calculate one element of `c` specifically at index `c[cy * n + cx] == c[cy][cx]`

Now consider an output row in `c`. Each element

$$
c_{i,j}, \qquad i \in \{0,1,\dots,m-1\}, \quad j \in \{0,1,\dots,n-1\}
$$

is calculated by taking the dot product between the entire row $a_{i,:}$ and the entire column $b_{:,j}$:

$$
c_{i,j} = a_{i,:} \cdot b_{:,j}.
$$

Thus, an entire row $c_{i,:}$ is calculated by taking the dot product between $a_{i,:}$ and $b_{:,j}$ for every $j \in \{0,1,\dots,n-1\}$. Similarly, an entire column $c_{:,j}$ is calculated by taking the dot product between $a_{i,:}$ and $b_{:,j}$ for every $i \in \{0,1,\dots,m-1\}$.

Therefore, each element of `a` is reused `n` times across the complete matrix multiplication, while each element of `b` is reused `m` times. The naive kernel repeatedly issues loads for the same elements of `a` and `b`. Rather than repeatedly relying on global memory accesses and the hardware cache hierarchy to provide this reused data, we can explicitly load portions of `a` and `b` into shared memory and repeatedly access them from there.

Now consider our shared memory strategy. Each thread computes one output element by walking along one row of `a` and one column of `b`. Scaling this idea to an entire `16 x 16` thread block, the block computes a `16 x 16` tile of `c`. Since the block computes a `16 x 16` tile of `c`, the block's output elements require 16 entire rows of `a` and 16 entire columns of `b`.

However, we do not need to load those entire rows and columns into shared memory at once. The dot products can be calculated incrementally along the `k` dimension. Since `TILE_K = 16`, we divide the dot products into chunks of 16 elements.

For each chunk, the block cooperatively loads:

- a `16 x 16` tile from `a`, containing 16 output rows and the next 16 columns along the `k` dimension
- a `16 x 16` tile from `b`, containing 16 output columns and the corresponding 16 rows along the `k` dimension

After these tiles are loaded into shared memory, each thread performs a partial dot product using the 16 values it needs from the two shared memory tiles. Once every thread in the block has finished using the current tiles, the block advances along the `k` dimension. It then loads the next 16 columns from `a` and the next 16 rows from `b` into shared memory, and each thread performs another partial dot product.

This process repeats until the block has traversed the entire `k` dimension. At that point, each thread's accumulator contains the complete dot product for its assigned output element in `c`.

The optimization comes from explicitly loading each tile from global memory and then reusing its values from shared memory across many threads. In the naive kernel, repeated accesses do not necessarily go all the way back to VRAM because the GPU's L1 and L2 caches may retain frequently accessed data. However, cache behavior is managed automatically by the hardware, so we do not directly control which values remain cached or how long they remain there. With shared memory, we explicitly choose which data to keep on chip and organize its reuse around the computation performed by the thread block. For an algorithm such as matrix multiplication, where the reuse pattern is known in advance, this explicit control can be more efficient and predictable than relying entirely on the cache hierarchy. This does not mean shared memory is always faster than an L1 cache hit. Rather, shared memory allows us to deliberately structure data reuse and avoid repeated global memory accesses instead of relying on the cache hierarchy to capture that reuse efficiently.

Now, take a look at the kernel above. Let's go through it step by step. First, we create two shared memory tiles for the input matrices `a` and `b`:

```cuda
__shared__ float aTile[BLOCK_Y][TILE_K];
__shared__ float bTile[TILE_K][BLOCK_X];
```

This reserves space in the SM's shared memory for the thread block. Since `BLOCK_X = BLOCK_Y = TILE_K = 16`, each tile contains `16 * 16 = 256` floats. Since each float is 4 bytes, each tile requires `256 * 4` bytes of shared memory.

Next, we get the thread's indices within the block and determine the output element of `c` that the thread is assigned to calculate:

```cuda
uint32_t ty = threadIdx.y;
uint32_t tx = threadIdx.x;

uint32_t cy = blockIdx.y * blockDim.y + ty;
uint32_t cx = blockIdx.x * blockDim.x + tx;
```

Next, we loop across the `k` dimension in chunks of `TILE_K = 16`:

```cuda
float acc = 0.0f;
for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {
    if (tx < TILE_K) {
        uint32_t ax = tileStart + tx;
        aTile[ty][tx] = (cy < m && ax < k) ? a[cy * k + ax] : 0.0f;
    }

    if (ty < TILE_K) {
        uint32_t by = tileStart + ty;
        bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
    }

    __syncthreads();

    for (uint32_t i = 0; i < TILE_K; i++) {
        acc += aTile[ty][i] * bTile[i][tx];
    }

    __syncthreads();
}
```

During each iteration, the thread block cooperatively loads one `16 x 16` tile from `a` and one `16 x 16` tile from `b` into shared memory. Each thread loads one element from `a` and one element from `b`. If the requested element falls outside the bounds of the matrix, the thread sets the corresponding shared memory element to `0.0f` instead.

The first `__syncthreads()` waits until every thread in the block has finished loading its elements into shared memory. This ensures that the complete tiles are available before any thread begins using them.

Each thread then performs a partial dot product using one row of `aTile` and one column of `bTile`:

```cuda
for (uint32_t i = 0; i < TILE_K; i++) {
    acc += aTile[ty][i] * bTile[i][tx];
}
```

Since `TILE_K = 16`, each thread performs `16 multiply-add` operations during each tile iteration, accumulating each product directly into `acc`.

The second  `__syncthreads()` waits until every thread in the block has finished using the current shared memory tiles before any thread begins overwriting them with the next tiles from `a` and `b`.

An easy way to think about this is that each iteration performs the same computation as a complete `16 x 16` matrix multiplication between the current `16 x 16` tile of `a` and the current `16 x 16` tile of `b`. If `a` and `b` were themselves only `16 x 16`, then this single tile iteration would be the complete matrix multiplication. For larger matrices, however, each tile multiplication contributes only a partial result to the block's `16 x 16` output tile of `c`. The block repeatedly loads and multiplies these tiles as it moves across the `k` dimension, accumulating the partial results until every thread has computed its complete dot product.

Finally, each thread writes the completed output value it was assigned to `c`:

```cuda
if (cy < m && cx < n) {
    c[cy * n + cx] = acc;
}
```

The next optimization shifts some of the work from thread level parallelism to per thread computation, allowing each thread to reuse loaded values across multiple outputs elements of `c`.

## Matrix Kernel Optimization 3: 1D Register Tiling

The idea behind 1D register tiling is to assign each thread multiple output elements instead of having each thread calculate only one. A thread will typically compute 4 or 8 output elements. The “1D” refers to the fact that these outputs are assigned along a single dimension of the output matrix.

In our case, each thread computes multiple rows within the same output column. Neighboring threads are still assigned neighboring output columns, which preserves the coalesced write pattern to the output matrix. At the same time, each thread can reuse the same value that was loaded into a register across several calculations.

At first, assigning more outputs to each thread appears to reduce parallelism. However, each thread can keep multiple partial sums in registers and reuse the same shared memory value across several multiply accumulate operations. This increases the arithmetic intensity with respect to shared memory, meaning that more arithmetic is performed for each byte read from shared memory. To be explicit, a thread loads a value from shared memory into a register that is needed by several of its output calculations, then reuses the element that occupies a register across those calculations. As a result, the kernel performs fewer shared memory reads relative to the amount of computation being performed.

As long as enough warps remain active to keep the GPU busy, the reduction in thread level parallelism can be outweighed by the increased data reuse and lower memory access overhead.

Below is the 1D register tiled kernel:

```cuda
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t ROWS_PER_THREAD = 8;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_K = 16;


__global__ void matMulCuda1DRegTileKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[TILE_M][TILE_K];
    __shared__ float bTile[TILE_K][BLOCK_X];

    uint32_t ty = threadIdx.y;
    uint32_t tx = threadIdx.x;

    uint32_t cyStart = blockIdx.y * TILE_M + ty * ROWS_PER_THREAD;
    uint32_t cx = blockIdx.x * blockDim.x + tx;

    float acc[ROWS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        // load aTile
        if (tx < TILE_K) {
            uint32_t aTileYStart = ty * ROWS_PER_THREAD;
            uint32_t ax = tileStart + tx;

            for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
                uint32_t aTileY = aTileYStart + r;
                uint32_t ay = cyStart + r;
                aTile[aTileY][tx] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
            }
        }

        // load bTile
        if (ty < TILE_K) {
            uint32_t by = tileStart + ty;
            bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
        }

        __syncthreads();

        // compute partial dot product
        #pragma unroll
        for (uint32_t i = 0; i < TILE_K; ++i) {
            float bVal = bTile[i][tx];

            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_THREAD; ++r) {
                acc[r] += aTile[ty * ROWS_PER_THREAD + r][i] * bVal;
            }
        }

        __syncthreads();
    }

    // write output values
    #pragma unroll
    for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
        uint32_t cy = cyStart + r;
        if (cy < m && cx < n) {
            c[cy * n + cx] = acc[r];
        }
    }
}
```

The input matrices are `a` and `b`, and the output matrix is `c`. Their dimensions are:

- `a`: `m` rows and `k` columns
- `b`: `k` rows and `n` columns
- `c`: `m` rows and `n` columns

Since `BLOCK_X = BLOCK_Y = 16`. Thus, each thread block contains `BLOCK_Y * BLOCK_X = 256` threads.

Each thread is assigned to calculate `ROWS_PER_THREAD = 8` elements of `c`, all within the same output column `cx`:

```text
c[(cyStart + 0) * n + cx] == c[cyStart + 0][cx]
c[(cyStart + 1) * n + cx] == c[cyStart + 1][cx]
...
c[(cyStart + 7) * n + cx] == c[cyStart + 7][cx]
```

At first, this kernel may look quite different from the shared memory kernel, but the overall structure is actually very similar. The main difference is that each thread now computes eight output elements of `c` instead of one.

The shared memory strategy remains the same. The block still moves through the k dimension in chunks of `TILE_K = 16`, cooperatively loading pieces of `a` and `b` into shared memory before computing partial dot products.

However, because each thread now computes eight output rows instead of one, the block computes more rows of `c` at once. With
`BLOCK_Y = 16` and `ROWS_PER_THREAD = 8` the block computes `16 * 8 = 128` output rows.

Therefore, instead of loading a `16 x 16` tile from `a`, the block now loads a `128 x 16` tile from a. The `b` tile remains `16 x 16`, because the block still computes 16 output columns.

Thus, the block computes a `128 x 16` block of output elements of `c`.

So during each iteration along the `k` dimension, the block loads:

- `aTile: 128 x 16`
- `bTile: 16 x 16`

Now we will go through the kernel step by step.

First, we create the two shared memory tiles:

```cuda
__shared__ float aTile[TILE_M][TILE_K];
__shared__ float bTile[TILE_K][BLOCK_X];

```

Since `TILE_M = BLOCK_Y * ROWS_PER_THREAD = 16 * 8 = 128` aTile has dimensions `128 x 16`. `bTile` has dimensions `16 x 16`. `aTile` is larger than in the previous shared memory kernel because each thread now computes eight output rows. Therefore, instead of loading one element of `aTile` per `k`-tile iteration as in the previous kernel, each thread now loads eight elements, one from each of the eight rows assigned to it.

Next, we get the thread indices within the block:

```cuda
uint32_t ty = threadIdx.y;
uint32_t tx = threadIdx.x;
```

Since the block is `16 x 16`, therefore `ty = 0, ..., 15`, and `tx = 0, ..., 15`.

We then calculate the first output row and output column assigned to the thread:

```cuda
uint32_t cyStart = blockIdx.y * TILE_M + ty * ROWS_PER_THREAD;
uint32_t cx = blockIdx.x * blockDim.x + tx;
```

The expression `blockIdx.y * TILE_M` moves to the first output row assigned to the current thread block. Within that block, each thread row is responsible for `ROWS_PER_THREAD = 8` output rows. Therefore, `ty * ROWS_PER_THREAD` skips over the groups of rows already assigned to previous threads within the same block. For example:

```text
ty = 0 → rows  0, ..., 7 within the block tile
ty = 1 → rows  8, ..., 15
ty = 2 → rows 16, ..., 23
...
ty = 15 → rows 120, ..., 127
```

Adding this offset to the starting row of the block gives the global starting row assigned to the thread.

The output column is calculated normally `uint32_t cx = blockIdx.x * blockDim.x + tx;`. Neighboring `tx` values therefore correspond to neighboring output columns, which preserves the coalesced write pattern when the final results are written to c.

Next, we allocate one accumulator for each output element assigned to the thread:

```cuda
float acc[ROWS_PER_THREAD] = {0.0f};
```

Since `ROWS_PER_THREAD = 8`, every thread maintains eight partial dot products:

```text
acc[0]
acc[1]
...
acc[7]
```

These values are intended to remain in registers while the kernel moves across the `k` dimension.

Next, we loop across the `k` dimension in chunks of `TILE_K = 16`:

```cuda
for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K)
```

The block first loads the current tile from `a`:

```cuda
if (tx < TILE_K) {
    uint32_t aTileYStart = ty * ROWS_PER_THREAD;
    uint32_t ax = tileStart + tx;

    for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
        uint32_t aTileY = aTileYStart + r;
        uint32_t ay = cyStart + r;
        aTile[aTileY][tx] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
    }
}
```

Since each thread is now responsible for `ROWS_PER_THREAD` output rows, each thread must also load one element from each of the `ROWS_PER_THREAD` rows into `aTile`. The variable `aTileYStart` calculates the first row of `aTile` that the current thread is responsible for loading. We multiply `ty` by `ROWS_PER_THREAD` because every previous thread row within the block has already been assigned `ROWS_PER_THREAD` rows of `aTile`.

For example, with `ROWS_PER_THREAD = 8`:

```text
ty = 0 → aTile rows  0..7
ty = 1 → aTile rows  8..15
ty = 2 → aTile rows 16..23
...
ty = 15 → aTile rows 120..127
```

The loop then loads those eight rows at the current `k`-dimension position `ax`:

```cuda
for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
    uint32_t aTileY = aTileYStart + r;
    uint32_t ay = cyStart + r;
    aTile[aTileY][tx] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
}
```

Thus, each thread loads eight elements from `a` into one column of `aTile`, and together the threads in the block cooperatively fill the complete `128 x 16` shared memory tile.

The block then loads the `16 x 16` tile of `b` exactly as in the previous shared memory kernel:

```cuda
if (ty < TILE_K) {
    uint32_t by = tileStart + ty;
    bTile[ty][tx] = (by < k && cx < n) ? b[by * n + cx] : 0.0f;
}
```

Once both shared memory tiles are loaded, the block synchronizes. This ensures that every element of `aTile` and `bTile` is available before any thread begins using the tiles.

Next, each thread computes eight partial dot products:

```cuda
for (uint32_t i = 0; i < TILE_K; ++i) {
    float bVal = bTile[i][tx];

    #pragma unroll
    for (uint32_t r = 0; r < ROWS_PER_THREAD; ++r) {
        acc[r] +=
            aTile[ty * ROWS_PER_THREAD + r][i] * bVal;
    }
}
```

This is where the main 1D register tiling optimization occurs. For each value of `i`, the thread loads one value from `bTile`:

```cuda
float bVal = bTile[i][tx];
```

That value is stored in a register and reused across all eight output rows assigned to the thread. Conceptually:

```text
aTile[row 0][i] * bVal → acc[0]
aTile[row 1][i] * bVal → acc[1]
aTile[row 2][i] * bVal → acc[2]
...
aTile[row 7][i] * bVal → acc[7]
```

Instead of loading the same value from `bTile` separately for eight different output calculations, the thread loads it once into `bVal` and reuses it across eight multiply-accumulate operations. This is the main source of additional reuse compared with the previous shared memory kernel. The accumulator array is intended to remain in registers throughout the computation. Unrolling the small loop helps the compiler access each accumulator using a constant index, although actual register placement depends on the compiled kernel.

Once all threads have finished using the current shared memory tiles, the second `__syncthreads()` ensures that no thread begins overwriting those tiles with the next `k` chunk until every thread has finished using them. The process then repeats for the next `TILE_K = 16` section of the dot products.

After the block has traversed the complete `k` dimension, each thread has eight completed output values stored in its accumulator array. Finally, the thread writes those eight values to `c`:

```cuda
#pragma unroll
for (uint32_t r = 0; r < ROWS_PER_THREAD; r++) {
    uint32_t cy = cyStart + r;

    if (cy < m && cx < n) {
        c[cy * n + cx] = acc[r];
    }
}
```

Each thread writes eight rows within one column:

```text
c[(cyStart + 0)][cx]
c[(cyStart + 1)][cx]
...
c[(cyStart + 7)][cx]
```

While neighboring threads write neighboring columns. For a fixed `r` and `ty`, in a block with `blockIdx.x = 0`, neighboring threads write:

```text
thread tx = 0 → c[cy][0]
thread tx = 1 → c[cy][1]
thread tx = 2 → c[cy][2]
...
```

Therefore, the global memory writes remain coalesced even though each thread now computes multiple output elements.

Next, we extend register tiling to two dimensions, assigning each thread multiple output rows and columns so it can reuse values from both `aTile` and `bTile` across several calculations.

## Matrix Kernel Optimization 4: 2D Register Tiling

The idea behind 2D register tiling directly builds off the previous optimization. In the 1D register tiled kernel, each thread calculated multiple output elements down a single column of the output matrix. We now extend this idea so that each thread calculates a small 2D patch of output elements instead. This allows us to further increase the arithmetic intensity with respect to shared memory.

You might be wondering what the difference is between 2D register tiling and simply extending the length of the 1D tile. As we will see in the 2D register tiled kernel, each thread loads multiple values from both shared memory tiles into registers and then uses the values that occupy the registers to compute a small patch of partial dot products. This reduces the total number of values needed from shared memory for the same number of multiply-accumulate operations. For example, at each position along `k`, a 16 element 1D tile needs 16 values from the left input matrix and one from the right input matrix. A `4 x 4` tile needs only four values from each input matrix. Both perform 16 multiply-accumulate operations, but the 2D tile needs eight input values instead of seventeen. In other words, we increase the arithmetic intensity with respect to shared memory even further by creating reuse in both dimensions rather than only one.

For example, in a 1D register tile, one value loaded from one input matrix may be reused across several output rows, while the corresponding value from the other input matrix contributes to only one output within that thread’s 1D tile. In a 2D register tile, multiple values from both input matrices are loaded into registers and reused across a small 2D output patch. This allows a small number of shared memory reads to produce many multiply-accumulate operations.

That is essentially the main difference between the 1D and 2D register tiled kernels. The 2D kernel increases data reuse from both input matrices and therefore further reduces the amount of shared memory traffic relative to the amount of computation being performed.

At first, assigning more output elements to each thread may seem like it reduces parallelism because fewer threads are needed to compute the same output tile. However, the GPU only has a fixed amount of hardware available to execute threads at any given time. In practice, a matrix multiplication launches far more threads and thread blocks than can execute simultaneously, so there is usually already enough parallel work available to keep the SMs occupied.

Once the available execution resources are fully utilized, creating even more fine grained thread level parallelism does not necessarily make the kernel faster. At that point, memory access and data movement can become the limiting factor. By giving each thread more work and reusing values already loaded into registers, 2D register tiling reduces the amount of memory traffic required for the same amount of computation.

So even though each thread performs more work, the GPU can still have many warps and thread blocks active across its SMs. As long as there are enough resident warps to keep the SMs occupied, the important optimization becomes making each active thread more efficient with the data it already has. This allows the execution units to spend more time performing arithmetic and less time waiting for data from shared or global memory.

Below is the 2D register tiled kernel:

```cuda
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_Y * BLOCK_X;
constexpr uint32_t ROWS_PER_THREAD = 4;
constexpr uint32_t COLS_PER_THREAD = 4;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 32;
constexpr uint32_t A_TILE_ROW_STRIDE = NUM_THREADS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = NUM_THREADS / TILE_N;


__global__ void matMulCuda2DRegTileKernel (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ float aTile[TILE_M][TILE_K];
    __shared__ float bTile[TILE_K][TILE_N];

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tid = ty * BLOCK_X + tx;

    const uint32_t aTileYStart = tid / TILE_K;
    const uint32_t bTileYStart = tid / TILE_N;

    const uint32_t aTileX = tid % TILE_K;
    const uint32_t bTileX = tid % TILE_N;

    const uint32_t ayBlockStart = blockIdx.y * TILE_M;
    const uint32_t bxBlockStart = blockIdx.x * TILE_N;

    const uint32_t bx = bxBlockStart + bTileX;

    const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
    const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;

    float aReg[ROWS_PER_THREAD];
    float bReg[COLS_PER_THREAD];
    float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            uint32_t aTileY = tileRowOffset + aTileYStart;
            uint32_t ay = ayBlockStart + aTileY;
            uint32_t ax = tileStart + aTileX;
            aTile[aTileY][aTileX] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
        }

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            uint32_t bTileY = tileRowOffset + bTileYStart;
            uint32_t by = tileStart + bTileY;
            bTile[bTileY][bTileX] = (by < k && bx < n) ? b[by * n + bx] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                aReg[i] = aTile[ty * ROWS_PER_THREAD + i][dotIdx];
            }

            #pragma unroll
            for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                bReg[j] = bTile[dotIdx][tx * COLS_PER_THREAD + j];
            }

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        uint32_t cy = cyStart + i;

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
            uint32_t cx = cxStart + j;

            if (cy < m && cx < n) {
                c[cy * n + cx] = acc[i][j];
            }
        }
    }
}
```

This kernel may look quite a bit more intimidating than the 1D register tiled kernel, but the underlying idea is essentially the same. The main differences are that each thread now computes a 2D patch of output elements instead of a 1D column of outputs, and the shared memory loading strategy has been generalized so that loading the tiles is independent of the output elements assigned to each thread. Let's go through the kernel step by step.

First, we create the shared memory tiles `aTile` and `bTile`:

```cuda
__shared__ float aTile[TILE_M][TILE_K];
__shared__ float bTile[TILE_K][TILE_N];
```

The shape of `aTile` is still determined by the number of output rows computed by the block and the number of elements processed along the `k` dimension during each tile iteration. Therefore `aTile` has dimensions `(BLOCK_Y * ROWS_PER_THREAD) x TILE_K = (16 * 4 ) x 32 = 64 x 32`. The main difference from the 1D register tiled kernel is the shape of `bTile`. In the 1D register tiled kernel, the block computed only `BLOCK_X` output columns, so `bTile` only needed `BLOCK_X` columns. In the 2D register tiled kernel, every thread now computes `COLS_PER_THREAD = 4` output columns. Since there are `BLOCK_X = 16` thread columns in the block, the block as a whole computes `TILE_N = BLOCK_X * COLS_PER_THREAD = 16 * 4 = 64` output columns. Therefore, the block requires 64 columns from `b` during each `k`-tile iteration, giving the dimensions of `bTile` to be `TILE_K x TILE_N = 32 x 64`. So each thread block now computes a `64 x 64` tile of the output matrix.

Next, we get the thread's 2D coordinates within the block and also calculate a flattened thread index `tid`:

```cuda
const uint32_t ty = threadIdx.y;
const uint32_t tx = threadIdx.x;
const uint32_t tid = ty * BLOCK_X + tx;
```

Since the block contains `BLOCK_Y * BLOCK_X = 16 * 16 = 256` threads, `tid` ranges from 0 to 255. The flattened thread index allows us to treat all 256 threads in the block as a single group when cooperatively loading the shared memory tiles. This is useful because the dimensions of `aTile` and `bTile` no longer directly match the dimensions of the thread block.

Next, we calculate the starting tile row and tile column assigned to each thread when loading `aTile` and `bTile`:

```cuda
const uint32_t aTileYStart = tid / TILE_K;
const uint32_t aTileX = tid % TILE_K;

const uint32_t bTileYStart = tid / TILE_N;
const uint32_t bTileX = tid % TILE_N;
```

These calculations convert the flattened thread index back into a row and column coordinate for each shared memory tile. Since `aTile` contains `TILE_K` columns, dividing by `TILE_K` gives the starting row and taking the remainder gives the column:

```text
aTile row    = tid / TILE_K
aTile column = tid % TILE_K
```

Similarly, since `bTile` contains `TILE_N` columns:

```text
bTile row    = tid / TILE_N
bTile column = tid % TILE_N
```

These values define the starting row and fixed column for the thread’s cooperative tile loads. The column remains fixed for that thread, while the row is advanced by a fixed stride to load the remaining elements assigned to it.

Next, we calculate the starting row of matrix `a` and the starting column of matrix `b` corresponding to the current thread block:

```cuda
const uint32_t ayBlockStart = blockIdx.y * TILE_M;
const uint32_t bxBlockStart = blockIdx.x * TILE_N;
```

`ayBlockStart` gives the first row of matrix `a` needed to compute the output rows handled by the current thread block. Since the block computes `TILE_M` output rows, multiplying `blockIdx.y` by `TILE_M` moves us to the corresponding starting row in `a`.

Similarly, `bxBlockStart` gives the first column of matrix `b` needed to compute the output columns handled by the current thread block. Since the block computes `TILE_N` output columns, multiplying `blockIdx.x` by `TILE_N` moves us to the corresponding starting column in `b`.

These values depend only on the block coordinates, so they remain constant for the entire lifetime of the block and can be calculated once before entering the k-tile loop.

Next, we calculate `bx`, which gives the global column of matrix `b` that this thread is responsible for loading:

```cuda
const uint32_t bx = bxBlockStart + bTileX;
```

`bx` can be calculated once before entering the `k`-tile loop because it depends only on `bxBlockStart` and `bTileX`, both of which remain constant for the lifetime of the thread. Unlike `by`, `bx` does not depend on `tileStart`, so its value does not change as we move through different `TILE_K` chunks.

Next, we calculate the starting output row and column of the output patch that this thread will compute:

```cuda
const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;
```

The row calculation is the same idea as in the 1D register tiled kernel. `ayBlockStart = blockIdx.y * TILE_M` is an index with respect to the output matrix, so it moves us to the first output row covered by the current thread block. From there, `ty * ROWS_PER_THREAD` moves to the first output row assigned to the current thread row within that block.

The column calculation follows the same pattern. `bxBlockStart = blockIdx.x * TILE_N` moves to the first output column covered by the current thread block, while `tx * COLS_PER_THREAD` moves to the first output column assigned to the current thread column within that block. Therefore, each thread begins at `(cyStart, cxStart)` and computes a `ROWS_PER_THREAD × COLS_PER_THREAD = 4 x 4` patch of the output matrix.

Next, we declare the registers used during the partial dot product calculations:

```cuda
float aReg[ROWS_PER_THREAD];
float bReg[COLS_PER_THREAD];
float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};
```

`aReg` stores the `ROWS_PER_THREAD = 4` values loaded from `aTile` for the current `dotIdx`, while `bReg` stores the `COLS_PER_THREAD = 4` values loaded from `bTile` for that same `dotIdx`. The 2D accumulator array `acc` stores the partial results for the thread's complete `4 x 4` output patch. Therefore, each thread maintains `4 x 4 = 16` partial output values throughout the traversal of the `k` dimension.

Next, we enter the main `k`-tile loop, where the block loads the current tiles from `a` and `b` into shared memory and then computes the corresponding partial dot products. Before looking at the partial dot product calculations, it is useful to first examine how the shared memory tiles are now loaded, since this loading strategy is more general than the one used in the 1D register tiled kernel.

```cuda
for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K)
```

In the 1D register tiled kernel, the shared memory loading pattern was directly tied to the thread coordinates used for the output computation. Although this produced coalesced global memory reads, it also meant that the loading strategy depended on the dimensions of the thread block and the output elements assigned to each thread. In the 2D register tiled kernel, we instead decouple shared memory loading from output computation. The flattened thread index `tid` is used to distribute the tile elements across all 256 threads independently of the output patch each thread computes. Another advantage of the new loading strategy is that the shared memory tile dimensions are no longer constrained by the individual dimensions of the thread block. In the previous loader, `tx` directly selected columns of `aTile` and `ty` directly selected rows of `bTile`, which meant `TILE_K` could not exceed the corresponding block dimensions without requiring a different loading scheme. By flattening the block into `tid`, we can instead use all `NUM_THREADS` threads cooperatively to cover the tiles. This allows us, for example, to use `TILE_K = 32` with a 16 x 16 thread block while still maintaining coalesced global memory accesses.

Below is the code responsible for loading the tiles:

```cuda
#pragma unroll
for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
    uint32_t aTileY = tileRowOffset + aTileYStart;
    uint32_t ay = ayBlockStart + aTileY;
    uint32_t ax = tileStart + aTileX;
    aTile[aTileY][aTileX] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
}

#pragma unroll
for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
    uint32_t bTileY = tileRowOffset + bTileYStart;
    uint32_t by = tileStart + bTileY;
    bTile[bTileY][bTileX] = (by < k && bx < n) ? b[by * n + bx] : 0.0f;
}
```

We will explain the loading of `aTile` first, and then briefly look at `bTile`, since they use the same general loading strategy. First, we have the following loop:

```cuda
for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE)
```

At first this may look slightly unintuitive. We could instead write:

```cuda
for (uint32_t aTileY = aTileYStart; aTileY < TILE_M; aTileY += A_TILE_ROW_STRIDE)
```

and this would express the exact same loading pattern. However, starting the loop at 0 makes the fixed iteration structure explicit. For example,

```text
tileRowOffset = 0; tileRowOffset < 64; tileRowOffset += 8
```

clearly executes eight times regardless of the thread. This works naturally with `#pragma unroll`, since the compiler sees loop bounds and an increment that are compile time constants. The alternative form can also potentially be unrolled, but the offset formulation makes the constant iteration pattern particularly clear.

Next, recall these constants:

```cuda
constexpr uint32_t A_TILE_ROW_STRIDE = NUM_THREADS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = NUM_THREADS / TILE_N;
```

Since `aTile` contains `TILE_K` elements per row, `NUM_THREADS / TILE_K` tells us how many complete rows of `aTile` all threads in the block collectively cover during one loading pass. Likewise, because `bTile` contains `TILE_N` elements per row, `NUM_THREADS / TILE_N` tells us how many complete rows of `bTile` are covered during one loading pass.

With the values used in this kernel:

```text
A_TILE_ROW_STRIDE = 256 / 32 = 8
B_TILE_ROW_STRIDE = 256 / 64 = 4
```

Therefore, one pass of all 256 threads covers eight complete rows of `aTile` or four complete rows of `bTile`.

We then use these values as row strides. Each thread keeps its tile column fixed while moving down by the row stride on each iteration. This moves the thread to the next set of rows that have not yet been covered by the block while preserving the contiguous access pattern between neighboring threads.

Inside the `aTile` loading loop we calculate:

```cuda
uint32_t aTileY = tileRowOffset + aTileYStart;
uint32_t ay = ayBlockStart + aTileY;
uint32_t ax = tileStart + aTileX;
```

`aTileY` gives the row within `aTile` that the thread is currently loading. It is calculated by adding the current row offset to the thread's starting tile row.

`ay` converts that shared memory row into the corresponding global row of matrix `a` by adding the block's starting row.

`ax` gives the global column of matrix `a` for the current `TILE_K` iteration. `tileStart` gives the beginning of the current chunk along the `k` dimension, and `aTileX` gives the thread's fixed column within that chunk.

Finally, we load the corresponding value from global memory into shared memory:

```cuda
aTile[aTileY][aTileX] = (ay < m && ax < k) ? a[ay * k + ax] : 0.0f;
```

The boundary check handles partial tiles at the edges of the matrices. If the global index lies outside the matrix, we write `0.0f` into shared memory instead.

Let's go through an example with actual numbers. Suppose we are looking at block 0, so:

```text
blockIdx.y = 0
blockIdx.x = 0
```

and:

```text
A_TILE_ROW_STRIDE = 256 / 32 = 8
B_TILE_ROW_STRIDE = 256 / 64 = 4
```

For `aTile`, the thread assignments are:

```text
thread 0 writes to   aTile[0][0],  aTile[8][0],  ..., aTile[56][0]
thread 1 writes to   aTile[0][1],  aTile[8][1],  ..., aTile[56][1]
...
thread 31 writes to  aTile[0][31], aTile[8][31], ..., aTile[56][31]
thread 32 writes to  aTile[1][0],  aTile[9][0],  ..., aTile[57][0]
thread 33 writes to  aTile[1][1],  aTile[9][1],  ..., aTile[57][1]
...
thread 255 writes to aTile[7][31], aTile[15][31], ..., aTile[63][31]
```

If we flatten the `64 x 32` tile into a one-dimensional array, the same pattern becomes:

```text
thread 0 writes to   aTile[0],   aTile[256], ..., aTile[1792]
thread 1 writes to   aTile[1],   aTile[257], ..., aTile[1793]
...
thread 31 writes to  aTile[31],  aTile[287], ..., aTile[1823]
thread 32 writes to  aTile[32],  aTile[288], ..., aTile[1824]
thread 33 writes to  aTile[33],  aTile[289], ..., aTile[1825]
...
thread 255 writes to aTile[255], aTile[511], ..., aTile[2047]
```

Now the loading pattern becomes easier to see. Within each loading pass, neighboring threads access neighboring elements, giving us coalesced global memory reads. Between loading passes, each thread keeps its tile column fixed and moves down by `A_TILE_ROW_STRIDE` rows.

In flattened memory, this means that each thread's next destination is exactly `NUM_THREADS = 256` elements after its previous destination. For example, thread 0 first writes flattened index `0`, while thread 255 writes index `255`. On the next pass, thread 0 writes index `256`, directly following the first set of 256 writes.

The loading strategy for `bTile` is exactly the same. Since `bTile` has `TILE_N = 64` columns, all 256 threads cover four complete rows per loading pass:

```text
B_TILE_ROW_STRIDE = 256 / 64 = 4
```

The thread assignments are:

```text
thread 0 writes to   bTile[0][0],  bTile[4][0],  ..., bTile[28][0]
thread 1 writes to   bTile[0][1],  bTile[4][1],  ..., bTile[28][1]
...
thread 63 writes to  bTile[0][63], bTile[4][63], ..., bTile[28][63]
thread 64 writes to  bTile[1][0],  bTile[5][0],  ..., bTile[29][0]
thread 65 writes to  bTile[1][1],  bTile[5][1],  ..., bTile[29][1]
...
thread 255 writes to bTile[3][63], bTile[7][63], ..., bTile[31][63]
```

Flattened, this becomes:

```text
thread 0 writes to  bTile[0],  bTile[256], ...,  bTile[1792]
thread 1 writes to  bTile[1],  bTile[257], ...,  bTile[1793]
...
thread 63 writes to  bTile[63],  bTile[319], ...,  bTile[1855]
thread 64 writes to  bTile[64],  bTile[320], ...,  bTile[1856]
thread 65 writes to  bTile[65],  bTile[321], ...,  bTile[1857]
...
thread 255 writes to  bTile[255],  bTile[511], ...,  bTile[2047]
```

An interesting result is that even though `aTile` and `bTile` have different two dimensional shapes, their flattened loading patterns are identical. In both cases, thread `tid` writes to:

```text
tid
tid + NUM_THREADS
tid + 2 * NUM_THREADS
...
```

The two dimensional coordinates differ because `aTile` and `bTile` have different row widths, but the flattened cooperative loading structure is the same.

Now that the shared memory tiles have been loaded, we synchronize the block to ensure that every thread has finished its loads before any thread begins reading from the tiles:

```cuda
__syncthreads();
```

We then load values from the shared memory tiles into registers and use those register values to compute the partial dot products:

```cuda
#pragma unroll
for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        aReg[i] = aTile[ty * ROWS_PER_THREAD + i][dotIdx];
    }

    #pragma unroll
    for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
        bReg[j] = bTile[dotIdx][tx * COLS_PER_THREAD + j];
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
            acc[i][j] += aReg[i] * bReg[j];
        }
    }
}
```

For each `dotIdx`, the thread loads `ROWS_PER_THREAD = 4` values from `aTile` into `aReg` and `COLS_PER_THREAD = 4` values from `bTile` into `bReg`. The values in `aReg` correspond to the four output rows assigned to the thread, while the values in `bReg` correspond to the four output columns assigned to the thread. We then combine every value in `aReg` with every value in `bReg`, producing:

```text
4 x 4 = 16
```

multiply-accumulate operations for the current `dotIdx`.

This is the key idea behind 2D register tiling. Instead of loading one value from shared memory and using it for only one output element, the values loaded into registers are reused across the thread's entire `4 x 4` output patch. Each value from `aReg` is reused across four output columns, while each value from `bReg` is reused across four output rows. After all `TILE_K = 32` positions have been processed, the thread has completed the partial dot products for the current pair of shared memory tiles. We then synchronize again:

```cuda
__syncthreads();
```

This second synchronization is needed before moving to the next `TILE_K` chunk. Without it, some threads could begin overwriting `aTile` and `bTile` with values from the next tile while other threads are still reading the current tile. The process then repeats for the next section of the `k` dimension until the complete dot products have been accumulated.

Finally, after all `TILE_K` chunks have been processed, each thread writes its `ROWS_PER_THREAD x COLS_PER_THREAD` output patch from the `acc` registers back to matrix `c`:

```cuda
#pragma unroll
for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
    uint32_t cy = cyStart + i;

    #pragma unroll
    for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
        uint32_t cx = cxStart + j;

        if (cy < m && cx < n) {
            c[cy * n + cx] = acc[i][j];
        }
    }
}
```

`cyStart` and `cxStart` give the top left output coordinate assigned to the thread. The loops then walk across the thread's `4 x 4` output patch, and each accumulated value is written to its corresponding position in `c`. The boundary check ensures that threads belonging to blocks along the edges of the matrix do not write outside the valid output dimensions.

With 2D register tiling established, the next optimization focuses on moving data with fewer instructions by combining four `float` values into one store/load operation. We do this by storing four floats in a single data type called `float4`.

## Matrix Kernel Optimization 5: Vectorization

Like the other optimizations thus far, vectorization is another incremental optimization. As mentioned in the previous paragraph, this involves combining four floats into a single data type called `float4`, which allows us to load and store four floats with a single instruction, as opposed to four separate instructions.

At first, this may seem quite similar to coalescing, since the entire point of coalescing was to execute a memory instruction using as few memory transactions as possible. If you remember, memory is read and written in 32 byte segments. So in a way, it may seem like we already have something similar to `float8` vectorization, since a 32 byte memory transaction can move eight floats at once, and then eight threads use those eight floats to do whatever work they need to do.

However, each thread still reads only one float per memory instruction. Vectorization allows each thread to read or write four contiguous floats with a single instruction. This is better explained with an example.

Assume we have a single warp, which, remember, contains 32 threads. Suppose that warp needs to read 128 contiguous floats from the float array `a`. Since a `float` is 4 bytes, the warp is responsible for reading:

```text
128 * 4 = 512 bytes
```

First, let's look at this as if we were going to use a regular `float` data type for reading these 128 floats. Since each memory transaction delivers 32 bytes and we have 512 bytes total, the minimum number of memory transactions possible is:

```text
512 / 32 = 16
```

which gives us perfect coalescing.

This can easily be done using the same general loading logic we used in the [2D register tiled kernel](#matrix-kernel-optimization-4-2d-register-tiling). We have 32 threads in our warp, so we increment `i` by 32 so that each iteration skips over the 32 reads already performed by the previous iteration.

```cuda
for (uint32_t i = tid; i < 128; i += 32) {
    float b = a[i];
    // calculations done...
}
```

Here, `tid` means the thread ID with respect to the warp, so `tid = 0, 1, ... 31`.

The access pattern looks like:

```text
Iteration 1:

thread 0: a[0]
thread 1: a[1]
...
thread 31: a[31]

Iteration 2:

thread 0: a[32]
thread 1: a[33]
...
thread 31: a[63]

...

Iteration 4:

thread 0: a[96]
thread 1: a[97]
...
thread 31: a[127]
```

So, as we can see, it takes four iterations to complete the task. At each iteration, the warp issues one load instruction for `a[i]`. Therefore, we have four warp-level load instructions. We also have perfect coalescing, as adjacent threads access adjacent elements, so we still hit our target of 16 memory transactions. Therefore, this code results in:

```text
Warp instructions:    4
Memory transactions: 16
```

Now let's use a `float4` loading strategy. The `float4` data type allows us to read or write four contiguous floats with one instruction. Since the floats are contiguous, each thread reads four contiguous floats from `a` per iteration. Because we have 32 threads, each reading four floats per iteration, the warp loads:

```text
4 * 32 = 128 floats
```

in a single iteration. This can be done using the following logic:

```cuda
for (uint32_t i = tid * 4; i < 128; i += 128) {
    float4 b = *reinterpret_cast<const float4*>(&a[i]);
    // calculations done...
}
```

For this `float4` load to be valid, `&a[i]` must be aligned to a 16 byte boundary, and the four floats beginning at `a[i]` must all lie within the allocation. Fortunately, CUDA allocations are aligned to at least 256 bytes, so we usually do not need to worry about the starting address of the array. We only need to ensure that we do not break this alignment by starting a `float4` load at the wrong offset. Since each `float4` contains four floats, the starting index must be a multiple of four, and all four values must remain within the array. In this example, `i` is always a multiple of four, so the loads are correctly aligned.

Again, `tid` means the thread ID with respect to the warp, so `tid = 0, 1, ... 31`.

The access pattern looks like:

```text
Iteration 1:

thread 0: a[0],   a[1],   a[2],   a[3]
thread 1: a[4],   a[5],   a[6],   a[7]
...
thread 31: a[124], a[125], a[126], a[127]
```

So, as we can see, we only have one iteration and therefore only one load instruction for the warp.

Again, we still have perfect coalescing because the threads within the warp read contiguous blocks of memory from `a`. `a[0]` through `a[7]` are covered by one 32 byte memory transaction and are used by threads 0 and 1. `a[8]` through `a[15]` are covered by another memory transaction and are used by threads 2 and 3. This pattern continues across the warp. Each 32 byte transaction contains eight floats, while each thread consumes four floats, so each transaction serves two adjacent threads. Therefore, we have `32 / 2 = 16 memory transactions`. Each thread now reads four floats using a single memory instruction. Therefore, this code results in:

```text
Warp instructions:    1
Memory transactions: 16
```

So, as we can see, the number of memory transactions remains the same, but the scalar `float` version requires four times as many warp-level memory instructions as the `float4` version. Memory instructions can overlap with other work, but the GPU still needs to issue and execute all four instructions, which introduces additional instruction overhead.

Therefore, using `float4` reads and writes can reduce the number of memory instructions the GPU needs to execute while preserving the same coalesced memory access pattern.

Now we move into the actual kernel, which is shown below:

```cuda
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_X * BLOCK_Y;

constexpr uint32_t ROWS_PER_THREAD = 8;
constexpr uint32_t COLS_PER_THREAD = 8;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 32;

constexpr uint32_t FLOATS_PER_FLOAT4 = 4;
constexpr uint32_t FLOATS_PER_LOAD_PASS = NUM_THREADS * FLOATS_PER_FLOAT4;

constexpr uint32_t A_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_N;

constexpr uint32_t BYTES_PER_FLOAT4 = 16;


__global__ void matMulCudaVectorizedExactFitKernel (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ __align__(BYTES_PER_FLOAT4) float aTile[TILE_K][TILE_M];
    __shared__ __align__(BYTES_PER_FLOAT4) float bTile[TILE_K][TILE_N];

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tid = ty * BLOCK_X + tx;
    const uint32_t threadStartIdx = tid * FLOATS_PER_FLOAT4;

    const uint32_t aTileYStart = threadStartIdx / TILE_K;
    const uint32_t bTileYStart = threadStartIdx / TILE_N;

    const uint32_t aTileX = threadStartIdx % TILE_K;
    const uint32_t bTileX = threadStartIdx % TILE_N;

    const uint32_t ayBlockStart = blockIdx.y * TILE_M;
    const uint32_t bxBlockStart = blockIdx.x * TILE_N;

    const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
    const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;

    __align__(BYTES_PER_FLOAT4) float aReg[ROWS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float bReg[COLS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            uint32_t aTileY = tileRowOffset + aTileYStart;
            uint32_t ay = ayBlockStart + aTileY;
            uint32_t ax = tileStart + aTileX;

            const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
            aTile[aTileX + 0][aTileY] = toLoad.x;
            aTile[aTileX + 1][aTileY] = toLoad.y;
            aTile[aTileX + 2][aTileY] = toLoad.z;
            aTile[aTileX + 3][aTileY] = toLoad.w;
        }

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            uint32_t bTileY = tileRowOffset + bTileYStart;
            uint32_t by = tileStart + bTileY;
            uint32_t bx = bxBlockStart + bTileX;

            *reinterpret_cast<float4*>(&bTile[bTileY][bTileX]) = *reinterpret_cast<const float4*>(&b[by * n + bx]);
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i += FLOATS_PER_FLOAT4) {
                *reinterpret_cast<float4*>(&aReg[i]) = *reinterpret_cast<float4*>(&aTile[dotIdx][ty* ROWS_PER_THREAD + i]);
            }

            #pragma unroll
            for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
                *reinterpret_cast<float4*>(&bReg[j]) = *reinterpret_cast<float4*>(&bTile[dotIdx][tx * COLS_PER_THREAD + j]);
            }

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        uint32_t cy = cyStart + i;

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
            uint32_t cx = cxStart + j;
            *reinterpret_cast<float4*>(&c[cy * n + cx]) = *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}
```

This is an exact-fit kernel, so it requires the matrix dimensions to be exactly divisible by the tile dimensions. Since

$$
TILE\_M = 128,\qquad TILE\_N = 128,\qquad TILE\_K = 32,
$$

we require

$$
m \bmod 128 = 0,\qquad
n \bmod 128 = 0,\qquad
k \bmod 32 = 0.
$$

A general version of this kernel is also included in the repository for matrix dimensions that are not exact multiples of the tile dimensions.

From here on out, I will stop going over the entire kernel line by line and only explain the parts that are new to this kernel.

First are these constants:

```cuda
constexpr uint32_t FLOATS_PER_FLOAT4 = 4;
constexpr uint32_t FLOATS_PER_LOAD_PASS = NUM_THREADS * FLOATS_PER_FLOAT4;

constexpr uint32_t A_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_N;

constexpr uint32_t BYTES_PER_FLOAT4 = 16;
```

`FLOATS_PER_FLOAT4 = 4` is fairly obvious from the name: there are four float values inside a `float4`.

Next, we have `FLOATS_PER_LOAD_PASS`. Remember from the earlier kernels that each thread block contains `BLOCK_X * BLOCK_Y = 16 * 16 = 256` threads. In the previous shared memory loading strategy, each thread loaded one `float` during a loading pass, so one pass of all 256 threads loaded 256 floats.

Now each thread loads a `float4`, meaning each thread loads four floats instead of one. Therefore, one loading pass across all threads moves:
`NUM_THREADS * FLOATS_PER_FLOAT4 = 256 * 4 = 1024 = FLOATS_PER_LOAD_PASS` floats.

The row strides `A_TILE_ROW_STRIDE` and `B_TILE_ROW_STRIDE` use the same logic as in the 2D register tiled kernel. The difference is that the numerator is now `FLOATS_PER_LOAD_PASS` instead of `NUM_THREADS`, because each loading pass moves 1024 floats instead of 256.

Finally:

```cuda
constexpr uint32_t BYTES_PER_FLOAT4 = 16;
```

is simply because each float is four bytes, and a `float4` contains four floats. Thus `4 floats * 4 bytes = 16 bytes`.

The next new piece is `__align__(BYTES_PER_FLOAT4)` which means 16 byte alignment. For example:

```cuda
__shared__ __align__(BYTES_PER_FLOAT4) float bTile[TILE_K][TILE_N];
```

ensures that the address to the first float in `bTile` is aligned to 16 bytes. That means the address is 0 or divisible by 16.

We want this because a `float4` access moves 16 bytes at once. The address used for a `float4` load or store therefore needs to be 16 byte aligned if we want to safely use that 16 byte access. In this kernel, the indices used for the `float4` accesses are also chosen in groups of four floats, so the addresses we use remain aligned to 16 byte boundaries.

Remember, the point of doing this is to allow the GPU to move four floats with one memory instruction instead of issuing four separate scalar load or store instructions.

The next new piece is:

```cuda
const uint32_t threadStartIdx = tid * FLOATS_PER_FLOAT4;
```

This serves a similar purpose to `tid` in the previous kernel, except now each thread is responsible for four contiguous floats during a loading pass instead of one. For example:

```text
thread 0 starts at float 0
thread 1 starts at float 4
thread 2 starts at float 8
...
```

We multiply `tid` by `FLOATS_PER_FLOAT4` so that each thread skips over the four floats assigned to the previous thread. This value is then used to calculate the starting row and column of the shared memory tile that each thread loads:

```cuda
const uint32_t aTileYStart = threadStartIdx / TILE_K;
const uint32_t bTileYStart = threadStartIdx / TILE_N;

const uint32_t aTileX = threadStartIdx % TILE_K;
const uint32_t bTileX = threadStartIdx % TILE_N;
```

The logic is the same as in the previous kernel, except that `threadStartIdx` now represents the first float of a four float group rather than the location of a single float.

The next important change is the layout of `aTile`:

```cuda
__shared__ __align__(BYTES_PER_FLOAT4) float aTile[TILE_K][TILE_M];
```

Compared with the previous kernel, the dimensions of `aTile` have been transposed. Previously we stored it as:

```cuda
aTile[TILE_M][TILE_K]
```

This changes how we load values into the shared memory tile:

```cuda
#pragma unroll
for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
    uint32_t aTileY = tileRowOffset + aTileYStart;
    uint32_t ay = ayBlockStart + aTileY;
    uint32_t ax = tileStart + aTileX;

    const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
    aTile[aTileX + 0][aTileY] = toLoad.x;
    aTile[aTileX + 1][aTileY] = toLoad.y;
    aTile[aTileX + 2][aTileY] = toLoad.z;
    aTile[aTileX + 3][aTileY] = toLoad.w;
}
```

We load four contiguous floats from global memory using a single `float4` load:

```cuda
const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
```

However, those four values are written individually into the transposed shared memory tile:

```cuda
aTile[aTileX + 0][aTileY] = toLoad.x;
aTile[aTileX + 1][aTileY] = toLoad.y;
aTile[aTileX + 2][aTileY] = toLoad.z;
aTile[aTileX + 3][aTileY] = toLoad.w;
```

At first, it may seem like directly writing the entire `float4` into shared memory would offer better kernel performance, because it requires fewer store instructions than writing the four values individually. For example, we could write the four values into `aTile` with a single `float4` store:

```cuda
*reinterpret_cast<float4*>(&aTile[aTileY][aTileX]) = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
```

This should indeed be faster. So why do we transpose `aTile` instead? The reason becomes clear when we look at how `aTile` is read during the partial dot product calculation:

```cuda
#pragma unroll
for (uint32_t i = 0; i < ROWS_PER_THREAD; i += FLOATS_PER_FLOAT4) {
    *reinterpret_cast<float4*>(&aReg[i]) = *reinterpret_cast<float4*>(&aTile[dotIdx][ty * ROWS_PER_THREAD + i]);
}
```

In the previous 2D register tiled kernel, the values needed for `aReg` were taken from different rows of `aTile` at the same `dotIdx`:

```cuda
aReg[i] = aTile[ty * ROWS_PER_THREAD + i][dotIdx];
```

Those values are separated by the row stride of the original `aTile` layout, so they are not contiguous in memory. Because of that, we cannot load four of them with a single `float4` access. By transposing `aTile`, those same values now lie next to each other in memory:

```cuda
aTile[dotIdx][ty * ROWS_PER_THREAD + i]
```

so four values can be loaded from shared memory into `aReg` with one `float4` load.

That is the entire reason for transposing `aTile`. We sacrifice the ability to write the loaded `float4` into shared memory as one contiguous `float4` store, but in return we make the values needed during the inner dot product loop contiguous, allowing us to vectorize the repeated shared memory reads.

This tradeoff is not automatically faster. We are making the global-memory-to-shared-memory write for `aTile` **not** vectorized in order to make the shared-memory-to-register reads during the inner loop vectorized. In my benchmarks, the transposed `aTile` version was faster than the version that kept the original layout.

The `bTile` side does not require this transpose because the values needed by each thread are already contiguous along the column dimension:

```cuda
*reinterpret_cast<float4*>(&bReg[j]) = *reinterpret_cast<float4*>(&bTile[dotIdx][tx * COLS_PER_THREAD + j]);
```

Therefore, `bTile` can use a direct `float4` store when loading from global memory.

Finally, the output values are also written back to `c` four floats at a time:

```cuda
*reinterpret_cast<float4*>(&c[cy * n + cx]) = *reinterpret_cast<float4*>(&acc[i][j]);
```

Since the output columns assigned to each thread are contiguous, four accumulated results can be written with one `float4` store.

That is essentially all of the new logic introduced by the vectorized kernel compared with the 2D register tiled kernel. The tiling strategy, register accumulation, and synchronization remain fundamentally the same. The main difference is that memory movement is now organized around groups of four contiguous floats so that we can reduce the number of load and store instructions executed by the GPU.

With the memory accesses now organized around groups of four contiguous floats, the next step is to look more closely at how these new shared memory access patterns behave and whether there are any inefficiencies left to remove.

## Matrix Kernel Optimization 6: Resolving Bank Conflicts

### Bank Conflicts

Before diving into this optimization, we need to go over some new terminology. First, we need to know what a shared memory bank is. Shared memory is divided into 32 banks that can serve memory accesses in parallel. Each bank is 4 bytes wide, and consecutive 4 byte chunks of shared memory are distributed across the banks. After bank 31 (0-indexed), the mapping wraps back to bank 0, as shown below:

```text
float array index:  0  1  2  3  ... 30  31  32  33  34 ...
bank:               0  1  2  3  ... 30  31   0   1   2 ...
```

The example above illustrates how each contiguous float is assigned to one of the 32 banks, and the banks wrap around from bank 31 to bank 0 every 128 bytes. Thus, an easy way to determine which bank a float index maps to is:

```text
bank = idx mod 32
```

Now that we know what a bank is and how to map an index to a bank, we need to explain what a bank conflict is. We know that many indices in an array map to the same bank. However, when each thread performs a 4 byte access, a bank can only serve one distinct 4 byte value at a time. This is best illustrated with an example. Consider the following instructions:

```cuda
uint32_t idx = tid * 32;
uint32_t element = aShared[idx];
```

where `tid` represents the threads ID. CUDA executes instructions at the warp level, so for this example consider the first warp, where `tid = 0, 1, ..., 31`.

```text
thread 0:  idx = 0 * 32  =   0. Accesses bank   0 mod 32 = 0
thread 1:  idx = 1 * 32  =  32. Accesses bank  32 mod 32 = 0
...
thread 31: idx = 31 * 32 = 992. Accesses bank 992 mod 32 = 0
```

As we can see, each thread in the warp accesses a different index, but all of those indices map to the same bank. Since the bank cannot serve all 32 different addresses simultaneously, these accesses must be serialized, meaning they are served one at a time rather than in parallel. Instead of servicing the warp's accesses in one conflict-free step, the hardware must perform 32 successive accesses to bank 0.

This is a bank conflict: threads in the same warp access different memory addresses that map to the same bank. If multiple threads in a warp access the same memory address in the same bank, that is okay because the value can be retrieved once and provided to all of those threads.

In the previous example, we looked at what happens when each thread in a warp accesses one `float`. Since there are 32 threads in a warp and each thread accesses one 4 byte `float`, the warp requests:

$$
32 \times 4 = 128 \text{ bytes}
$$

This matches the maximum amount of data that the 32 shared memory banks can serve in a single memory transaction:

$$
32 \text{ banks} \times 4 \text{ bytes per bank} = 128 \text{ bytes}
$$

But what happens if a warp instruction requests more than 128 bytes? Since shared memory can serve at most 128 bytes in a single transaction, the request must be split into multiple transactions. For example, consider the following instructions:

```cuda
uint32_t idx = tid * 4;
float4 fourFloats = *reinterpret_cast<float4*>(&aShared[idx]);
```

where `tid` represents the thread ID. CUDA executes instructions at the warp level, so for this example consider the first warp, where `tid = 0, 1, ..., 31`.

```text
thread 0:  idx = 0 * 4 =    0. Accesses banks     (0-3) mod 32 =   (0-3)
thread 1:  idx = 1 * 4 =    4. Accesses banks     (4-7) mod 32 =   (4-7)
thread 2:  idx = 2 * 4 =    8. Accesses banks    (8-11) mod 32 =  (8-11)
...
thread 7:  idx = 7 * 4 =   28. Accesses banks   (28-31) mod 32 = (28-31)

thread 8:  idx = 8 * 4 =   32. Accesses banks   (32-35) mod 32 =   (0-3)
thread 9:  idx = 9 * 4 =   36. Accesses banks   (36-39) mod 32 =   (4-7)
...
thread 31: idx = 31 * 4 = 124. Accesses banks (124-127) mod 32 = (28-31)
```

Each thread now accesses four banks because each `float4` contains four 4 byte float values. Across the entire warp, the instruction requests:

$$
32 \times 16 = 512 \text{ bytes}
$$

At first, this may look like a bank conflict. For example, threads 0, 8, 16, and 24 all access banks 0 through 3, but they access different addresses within those banks. However, there is one important detail missing. A bank conflict is not determined by comparing every bank accessed by the entire warp wide request at once. Instead, bank conflicts are determined within each shared memory transaction. Since a single shared memory transaction can serve at most 128 bytes, a 512 byte request requires at least:

$$
\dfrac{512 \text{ bytes}}{128 \text{ bytes per memory transaction}}=4 \text{ memory transactions}
$$

even when there are no bank conflicts. For the access pattern above, those four transactions correspond to:

```text
transaction 1: threads  0-7  → 128 bytes → banks 0-31
transaction 2: threads  8-15 → 128 bytes → banks 0-31
transaction 3: threads 16-23 → 128 bytes → banks 0-31
transaction 4: threads 24-31 → 128 bytes → banks 0-31
```

Within each transaction, every bank is accessed once. Therefore, each transaction can be served without any additional serialization. The fact that bank 0 is used again in a later transaction does not constitute a bank conflict.

More precisely, a bank conflict occurs when the shared memory access pattern requires additional serialization beyond the minimum number of transactions required to serve the warp instruction. This happens when multiple accesses in the same transaction hit the same bank at different addresses. Therefore, when a warp requests more than 128 bytes, we should not analyze bank conflicts across the entire warp wide request at once. Instead, we analyze the bank accesses within each 128 byte transaction independently.

For the access patterns considered so far, this gives us a simple rule of thumb. For each shared memory instruction, if each thread accesses one `float` address, threads in the warp should ideally access all 32 banks. If each thread accesses one unique `float4` address, the warp request is split into contiguous groups of eight threads, so within each group, accesses should ideally cover all 32 banks without any bank being used twice.

Now that we've found the new access pattern we want, we need a way to actually create it in shared memory. One way to remove bank conflicts is to use a logical-to-physical mapping. Instead of storing each logical tile coordinate directly at the same physical location in shared memory, we map it to a different physical coordinate that gives us the bank-access pattern we want.

### Fixing `aTile` Bank Conflicts

So, we need to figure out how to map each logical coordinate $(x,y)$ to a physical coordinate $(x',y')$. This allows us to keep the same logical layout of aTile, while changing where each value is physically stored in shared memory to remove the bank conflicts.

Let's go through how we can derive this mapping. More formally, we eventually need this mapping to be a **bijection**, meaning that it is both one-to-one and onto. One-to-one means that distinct logical coordinates map to distinct physical coordinates, while onto means that every physical coordinate gets mapped to by some logical coordinate.

We can define our mapping as

$$
F:L\rightarrow P
$$

where $L$ is the set of logical coordinates and $P$ is the set of physical coordinates.

Since logically `aTile` has 32 values in the $x$ direction and 128 values in the $y$ direction,

$$
L=\{0,\ldots,31\}\times\{0,\ldots,127\}.
$$

After the transpose, the physical coordinate space is

$$
P=\{0,\ldots,127\}\times\{0,\ldots,31\}.
$$

Both spaces contain

$$
32\cdot128=4096
$$

coordinates.

We will prove that our final transformation is a bijection after deriving it.

First, if we remember from the [vectorized kernel](#matrix-kernel-optimization-5-vectorization), we transposed `aTile`, so the first step in our function is to flip $(x,y)$.

Function after adding step 1:

$$
F(x,y)=(y,x)
$$

Remember that when we write a coordinate as $(x,y)$, the actual array access is

```cuda
aTile[y][x]
```

because arrays are indexed as `[row][column]`, or `[y][x]`.

For example, the logical coordinate

$$
(4,0)
$$

gets transformed into the physical coordinate

$$
(0,4),
$$

which corresponds to

```cuda
aTile[4][0]
```

in memory.

The next step is to shift the transposed $x$ coordinate to the correct bank. I have provided the bank mapping for each thread below, along with its `aTileX` and `aTileY` values:

| tid | aTileY | aTileX | Original Bank | Desired Bank | Bank Mapping |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 0  | 0 | 0  | 0 | 0  | 0 → 0 |
| 1  | 0 | 4  | 0 | 4  | 0 → 4 |
| 2  | 0 | 8  | 0 | 8  | 0 → 8 |
| 3  | 0 | 12 | 0 | 12 | 0 → 12 |
| 4  | 0 | 16 | 0 | 16 | 0 → 16 |
| 5  | 0 | 20 | 0 | 20 | 0 → 20 |
| 6  | 0 | 24 | 0 | 24 | 0 → 24 |
| 7  | 0 | 28 | 0 | 28 | 0 → 28 |
| 8  | 1 | 0  | 1 | 1  | 1 → 1 |
| 9  | 1 | 4  | 1 | 5  | 1 → 5 |
| 10 | 1 | 8  | 1 | 9  | 1 → 9 |
| 11 | 1 | 12 | 1 | 13 | 1 → 13 |
| 12 | 1 | 16 | 1 | 17 | 1 → 17 |
| 13 | 1 | 20 | 1 | 21 | 1 → 21 |
| 14 | 1 | 24 | 1 | 25 | 1 → 25 |
| 15 | 1 | 28 | 1 | 29 | 1 → 29 |
| 16 | 2 | 0  | 2 | 2  | 2 → 2 |
| 17 | 2 | 4  | 2 | 6  | 2 → 6 |
| 18 | 2 | 8  | 2 | 10 | 2 → 10 |
| 19 | 2 | 12 | 2 | 14 | 2 → 14 |
| 20 | 2 | 16 | 2 | 18 | 2 → 18 |
| 21 | 2 | 20 | 2 | 22 | 2 → 22 |
| 22 | 2 | 24 | 2 | 26 | 2 → 26 |
| 23 | 2 | 28 | 2 | 30 | 2 → 30 |
| 24 | 3 | 0  | 3 | 3  | 3 → 3 |
| 25 | 3 | 4  | 3 | 7  | 3 → 7 |
| 26 | 3 | 8  | 3 | 11 | 3 → 11 |
| 27 | 3 | 12 | 3 | 15 | 3 → 15 |
| 28 | 3 | 16 | 3 | 19 | 3 → 19 |
| 29 | 3 | 20 | 3 | 23 | 3 → 23 |
| 30 | 3 | 24 | 3 | 27 | 3 → 27 |
| 31 | 3 | 28 | 3 | 31 | 3 → 31 |

You might be wondering why I included the `aTileX` and `aTileY` values. Which is fair, but look closely at the pattern that emerges. The desired bank is simply

$$
\text{Desired Bank}=\text{aTileX}+\text{aTileY}.
$$

Now we need to connect this bank pattern back to our physical coordinate. Remember that

```cuda
float aTile[32][128];
```

is stored in row-major order. Therefore, for a physical coordinate $(x',y')$, the flat index is

$$
I=128y'+x'.
$$

Since shared memory has 32 banks, the bank for a `float` is

$$
B=I\bmod32.
$$

Substituting the flat index gives

$$
B=(128y'+x')\bmod32.
$$

Since 128 is a multiple of 32,

$$
128y'\bmod32=0,
$$

so this simplifies to

$$
B=x'\bmod32.
$$

Therefore, for `aTile`, the physical $x$ coordinate determines which bank the value maps to.

From the table above, we know that for the `toLoad.x` instruction we want

$$
x'=\text{aTileY}+\text{aTileX}.
$$

Using our logical coordinates, this is

$$
x'=y+x.
$$

The physical $y$ coordinate is still just the transposed logical $x$ coordinate:

$$
y'=x.
$$

Function after adding step 2:

$$
F(x,y)=(y+x,x)
$$

For the specific `toLoad.x` store we are analyzing, the values of $x$ and $y$ have a very useful structure because each thread loads a `float4`. Since each `float4` contains four floats, the starting logical $x$ coordinate for each thread advances by four:

```text
x: 0, 4, 8, 12, 16, 20, 24, 28
```

Within each thread group, $x$ therefore tells us the thread's position in that group, in steps of four banks. Meanwhile, $y$ identifies which of the four thread groups the thread belongs to:

```text
group 0 → y = 0
group 1 → y = 1
group 2 → y = 2
group 3 → y = 3
```

This is why adding them is useful. The $x$ value spaces accesses four banks apart, while $y$ fills in the four banks between those multiples of four. Therefore,

$$
x' = x + y
$$

gives:

```text
y = 0:  0, 4,  8, 12, 16, 20, 24, 28
y = 1:  1, 5,  9, 13, 17, 21, 25, 29
y = 2:  2, 6, 10, 14, 18, 22, 26, 30
y = 3:  3, 7, 11, 15, 19, 23, 27, 31
```

Together, these are all 32 banks exactly once, so the bank conflicts are removed for this instruction. However, this transformation was derived only from the `toLoad.x` store. We need a transformation that works for **all logical coordinates** in the tile, including the coordinates used by `toLoad.y`, `toLoad.z`, and `toLoad.w`. Here are the current instructions:

```cuda
const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);

aTile[aTileX + 0][aTileY] = toLoad.x;
aTile[aTileX + 1][aTileY] = toLoad.y;
aTile[aTileX + 2][aTileY] = toLoad.z;
aTile[aTileX + 3][aTileY] = toLoad.w;
```

After the transpose, this can be thought of as:

```cuda
aTile[aTilePhysicalY + 0][aTilePhysicalX] = toLoad.x;
aTile[aTilePhysicalY + 1][aTilePhysicalX] = toLoad.y;
aTile[aTilePhysicalY + 2][aTilePhysicalX] = toLoad.z;
aTile[aTilePhysicalY + 3][aTilePhysicalX] = toLoad.w;
```

So as we can see, with our `float4` load from `a`, the four values are stored in different rows of `aTile`, but in the same column. This is important because we want to preserve the transposed layout of these groups of four values. It also keeps the values arranged correctly for the aligned `float4` shared memory reads that we perform later in the kernel. For example, if `aTileX = 0`, the logical values are:

```cuda
aTile[0][aTileY] = toLoad.x;
aTile[1][aTileY] = toLoad.y;
aTile[2][aTileY] = toLoad.z;
aTile[3][aTileY] = toLoad.w;
```

After the transpose, we want them to remain in the same physical column:

```text
aTile[0][aTilePhysicalX] = toLoad.x
aTile[1][aTilePhysicalX] = toLoad.y
aTile[2][aTilePhysicalX] = toLoad.z
aTile[3][aTilePhysicalX] = toLoad.w
```

However, if we apply our current transformation

$$
F(x,y)=(y+x,x)
$$

independently to all four logical coordinates, we get:

```text
aTile[0][aTileY + 0] = toLoad.x
aTile[1][aTileY + 1] = toLoad.y
aTile[2][aTileY + 2] = toLoad.z
aTile[3][aTileY + 3] = toLoad.w
```

Now each of the four values has been shifted into a different physical column. The problem is that the raw logical $x$ coordinate changes for each of the four values. Instead, we want all four logical $x$ coordinates belonging to the same `float4` group to use the **same column shift**. The pattern we want is:

```text
aTileX:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15 | ...
shift:   0  0  0  0 | 4  4  4  4 | 8  8  8  8 | 12 12 12 12 | ...
```

In other words, we need to round $x$ down to the beginning of its group of four. This can be done with

$$
s=x-(x\bmod4),
$$

where $s$ denotes the shift. For example,

$$
x=6
$$

gives

$$
s=6-(6\bmod4)=6-2=4,
$$

while

$$
x=11
$$

gives

$$
s=11-(11\bmod4)=11-3=8.
$$

Thus, the shift changes from simply

$$
x
$$

to

$$
x-(x\bmod4).
$$

Function after adding step 3:

$$
F(x,y)= \left( y+x-(x\bmod4), x \right)
$$

At this point we have essentially found our transformation, but we are missing one thing. `aTile` has 128 columns, meaning the valid physical $x$ coordinates are

$$
0,\ldots,127.
$$

However, our transformed $x$ coordinate can go outside this range. The largest possible logical $y$ value is

$$
127,
$$

and the largest possible shift is

$$
28.
$$

Therefore, the transformed $x$ coordinate can be as large as

$$
127+28=155.
$$

To keep the physical $x$-coordinate inside the 128 columns of aTile, we wrap the transformed $x$-coordinate back into the valid range 0 to 127 using modulo 128.

$$
x'= \left( y+x-(x\bmod4) \right)\bmod128.
$$

Therefore, our final transformation is

$$
\boxed{ F(x,y)= \left( \left(y+x-(x\bmod4)\right)\bmod128, x \right) }
$$

There is one other useful property here. Taking the result modulo 128 does **not** change which bank the coordinate maps to, because 128 is itself a multiple of the 32 banks. For example,

$$
155\bmod128=27,
$$

but

$$
155\bmod32=27
$$

and

$$
27\bmod32=27.
$$

So the modulo 128 keeps the physical coordinate inside `aTile` while preserving the bank mapping that we designed. This gives us our final logical-to-physical coordinate transformation.

Next, we show that $F$ is well-defined as a function

$$
F:L\rightarrow P.
$$

Let $(x,y)\in L$. Then $x\in\{0,1,\dots,31\}$ and $y\in\{0,1,\dots,127\}$. The physical $x$ coordinate is $x'=\left(y+x-(x\bmod4)\right)\bmod128$. By definition of modulo 128 and since $y+x-(x\bmod4)\geq0$, we have $x'\in\{0,1,\dots,127\}$. The physical $y$ coordinate is $y'=x$. Since $x\in\{0,1,\dots,31\}$, we have $y'\in\{0,1,\dots,31\}$. Therefore, $F(x,y)\in P$.

Since the formula for $F$ gives exactly one ordered pair for every $(x,y)\in L$, $F$ is well-defined as a function

$$
F:L\rightarrow P.
$$

Now we prove that $F$ is a bijection. First, we prove that $F$ is one-to-one. Assume that $F$ is not one-to-one. Then there exist $(x_1,y_1),(x_2,y_2)\in L$ such that $(x_1,y_1)\neq(x_2,y_2)$ and $F(x_1,y_1)=F(x_2,y_2)$.

Now,

```math
F(x_1,y_1)=F(x_2,y_2)
\Leftrightarrow
((y_1+x_1-(x_1\bmod4))\bmod128, x_1)
=
((y_2+x_2-(x_2\bmod4))\bmod128, x_2)
```

which gives,

$$
\begin{aligned}
(y_1+x_1-(x_1\bmod4))\bmod128
&=
(y_2+x_2-(x_2\bmod4))\bmod128 \\
x_1 &= x_2
\end{aligned}
$$

Let $x$ denote the common value of $x_1$ and $x_2$. Then,

```math
(y_1+x_1-(x_1\bmod4))\bmod128
=
(y_2+x_2-(x_2\bmod4))\bmod128
\Leftrightarrow
(y_1+x-(x\bmod4))\bmod128
=
(y_2+x-(x\bmod4))\bmod128.
```

Let $c = x-(x\bmod4)$. Then,

```math
(y_1+c)\bmod128
=
(y_2+c)\bmod128.
```

Thus, when dividing $y_1+c$ and $y_2+c$ by 128, they produce the same remainder

$$
r\in\{0,1,\dots,127\}.
$$

By the Division Algorithm, there exist integers $q_1,q_2\in\mathbb Z$ such that

$$
\begin{aligned}
y_1+c&=128q_1+r \\
y_2+c&=128q_2+r.
\end{aligned}
$$

Solving both equations for $r$ gives

$$
\begin{aligned}
r&=y_1+c-128q_1 \\
r&=y_2+c-128q_2.
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
y_1 + c- 128q_1 &= y_2 + c - 128q_2 \\
y_1 + c - y_2 - c &= 128q_1 - 128q_2 \\
y_1 - y_2 &= 128(q_1 - q_2) \quad \text{where } (q_1 - q_2) \in \mathbb{Z} \\
y_1 - y_2 &= 128k \quad \text{where } k = q_1 - q_2.
\end{aligned}
$$

Since $L= \{0,1,\dots,31\} \times \{0,1,\dots,127\}$, therefore $y_1,y_2\in\{0,1,\dots,127\}$. Thus, $(y_1 - y_2) \in \{-127, -126, \dots, -1, 0, 1, \dots 127\}$. Therefore, it must be that $k = 0$, meaning

$$
y_1 - y_2 = 128k \Leftrightarrow y_1 = y_2.
$$

Thus, it has been shown that $x_1 = x_2$ and $y_1 = y_2$, meaning $(x_1, y_1) = (x_2, y_2)$ which contradicts our assumption that $(x_1,y_1)\neq(x_2,y_2)$. Therefore, $F$ must be one-to-one.

Now, we must prove that $F$ is onto.

Since $F$ is one-to-one, by definition $F$ maps distinct elements of $L$ to distinct elements in $F(L)$. Therefore $|F(L)| = |L|$.

Since $|L| = |P|$, therefore $|F(L)| = |P|$.

By definition $F(L) \subseteq P$.

Since $P$ is finite, any proper subset of $P$ must have strictly fewer elements than $P$. However, $|F(L)| = |P|$. Therefore, $F(L)$ cannot be a proper subset of $P$, so $F(L) = P$, proving that $F$ is onto.

Since $F$ is one-to-one and onto, by definition $F$ is a bijection.

Since we have proved that our logical-to-physical mapping is a bijection, we can be sure that every unique logical coordinate of `aTile` maps to exactly one unique physical coordinate, and every physical coordinate corresponds to exactly one logical coordinate.

Thus, we can rearrange the physical layout of `aTile` to produce the bank-access pattern we want without losing or overwriting any values.

Therefore, we have removed the bank conflicts when writing to `aTile`.

### Fixing `bTile` Bank Conflicts

First, let's analyze the current accesses to `bTile` from the [vectorized kernel](#matrix-kernel-optimization-5-vectorization) in this code:

```cuda
#pragma unroll
for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
    *reinterpret_cast<float4*>(&bReg[j]) = *reinterpret_cast<float4*>(&bTile[dotIdx][tx * COLS_PER_THREAD + j]);
}
```

We will freeze `dotIdx = 0`. I have made a table of the original and desired banks below for the first warp.

Since we are reading a `float4` from `bTile`, each thread accesses four banks. As we discussed earlier, the warp-wide request is split into contiguous groups of eight threads, so it is easier to analyze threads 0–7 rather than the entire warp at once.

First, we will look at the first iteration, where `j = 0`:

| tid | tx | `tx * COLS_PER_THREAD + j` | Original Banks | Desired Banks | Bank Mapping |
| ---: | ---: | ---: | :--- | :--- | :--- |
| 0 | 0 | 0  | 0–3   | 0–3   | 0–3 → 0–3 |
| 1 | 1 | 8  | 8–11  | 8–11  | 8–11 → 8–11 |
| 2 | 2 | 16 | 16–19 | 16–19 | 16–19 → 16–19 |
| 3 | 3 | 24 | 24–27 | 24–27 | 24–27 → 24–27 |
| 4 | 4 | 32 | 0–3   | 4–7   | 0–3 → 4–7 |
| 5 | 5 | 40 | 8–11  | 12–15 | 8–11 → 12–15 |
| 6 | 6 | 48 | 16–19 | 20–23 | 16–19 → 20–23 |
| 7 | 7 | 56 | 24–27 | 28–31 | 24–27 → 28–31 |

Now, the second iteration, where `j = 4`:

| tid | tx | `tx * COLS_PER_THREAD + j` | Original Banks | Desired Banks | Bank Mapping |
| ---: | ---: | ---: | :--- | :--- | :--- |
| 0 | 0 | 4  | 4–7   | 4–7   | 4–7 → 4–7 |
| 1 | 1 | 12 | 12–15 | 12–15 | 12–15 → 12–15 |
| 2 | 2 | 20 | 20–23 | 20–23 | 20–23 → 20–23 |
| 3 | 3 | 28 | 28–31 | 28–31 | 28–31 → 28–31 |
| 4 | 4 | 36 | 4–7   | 0–3   | 4–7 → 0–3 |
| 5 | 5 | 44 | 12–15 | 8–11  | 12–15 → 8–11 |
| 6 | 6 | 52 | 20–23 | 16–19 | 20–23 → 16–19 |
| 7 | 7 | 60 | 28–31 | 24–27 | 28–31 → 24–27 |

As we can see, in the `Desired Banks` column every bank from 0 through 31 is accessed exactly once within each 8-thread group, meaning there are no bank conflicts.

The pattern is also fairly simple. For `tid` 0–3, we leave the mapping unchanged. For `tid` 4–7, we swap the four-bank group accessed between the `j = 0` and `j = 4` iterations.

For example, for `tid = 4`:

```text
j = 0:  banks 0–3  → banks 4–7
j = 4:  banks 4–7  → banks 0–3
```

So rather than simply shifting all of the accesses in one direction, we are effectively **swapping the two groups of four values within each group of eight values**.

Now that we know the physical access pattern we want, we can derive a logical-to-physical mapping that produces it.

For `bTile`, the bank conflicts come entirely from which columns of `bTile` the `float4` reads access. This means our mapping only needs to affect $x$, while $y$ can stay the same. Since `COLS_PER_THREAD = 8`, each thread is responsible for eight output columns. Therefore, for a given `tx`, the logical `bTile` $x$ coordinates used by that thread are

$$
8tx, 8tx+1, \dots, 8tx+7.
$$

This naturally divides the logical $x$ coordinates of `bTile` into groups of eight consecutive values. Within each group of eight logical `bTile` columns, the two loop iterations access two separate `float4`s. When `j = 0`, the thread reads the first four logical columns in the group:

$$
8tx, 8tx+1, 8tx+2, 8tx+3.
$$

When `j = 4`, the thread reads the next four logical columns:

$$
8tx+4, 8tx+5, 8tx+6, 8tx+7.
$$

Now, look at the logical $x$ coordinates in groups of eight:

```text
groupIdx:      0        1        2       3       4       5       6       7
logical x:   0–7     8–15    16–23   24–31   32–39   40–47   48–55   56–63
action:     keep     keep     keep    keep    swap    swap    swap    swap
```

Now that we know our rule for when we want to swap the two `float4` groups within our group of 8 logical $x$ values, we need to express this rule mathematically.

We have 8 groups of 8 floats, where the first 4 groups are `keep` and the second 4 groups are `swap`. So, for these first 8 groups, our rule is to `swap` if

$$
\text{groupIdx} \geq 4.
$$

Now we just need to determine the group index for a given logical $x$ coordinate. Since each group contains 8 consecutive values, the group index is

$$
\left\lfloor \dfrac{x}{8} \right\rfloor.
$$

For example,

$$
x \in \{0,\dots,7\} \Rightarrow \text{groupIdx}=0,
$$

$$
x \in \{8,\dots,15\} \Rightarrow \text{groupIdx}=1,
$$

and so on.

This works for the first 8 groups, but what about group indices larger than 7? Since our `keep, keep, keep, keep, swap, swap, swap, swap` pattern repeats every 8 groups, we need to map the group index back into the range 0–7. We can do this using the $\bmod$ operator. This gives us the rule to `swap` if

$$
\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4
$$

Now, that we know how to determine if we need to swap an $x$ value, we need to figure out how to define an operation on $x$ to give us the desired swap. Consider $x = 38$, which from above we see is in `groupIdx = 4`, so we `swap` iteration 1 and iteration 2. Below is what we currently have:

```text
idx within group:  0   1  2  3 |  4  5  6  7
logical x:         32 33 34 35 | 36 37 38 39
```

and this is what we want:

```text
idx within group:  0   1  2  3 |  4  5  6  7
logical x:         36 37 38 39 | 32 33 34 35
```

For $x=38$, its index within the group is $38\bmod8=6.$ So before the swap, $x=38$ is at index 6 within the group. Looking at our desired layout, after swapping the two `float4` groups, $x=38$ should instead be at index 2:

```text
before: index 6
after:  index 2
```

More generally, the swap we want is
original idx:  0 1 2 3 | 4 5 6 7
swapped idx:   4 5 6 7 | 0 1 2 3

So we need an operation that maps

$$
0 \rightarrow 4, 1 \rightarrow 5, 2 \rightarrow 6, 3 \rightarrow 7,
$$

and

$$
4 \rightarrow 0, 5 \rightarrow 1, 6 \rightarrow 2, 7 \rightarrow 3.
$$

A simple way to get this is to add 4 to the `original idx` and then wrap the result back into the range 0–7 using modulo 8:

$$
\text{swapped idx} = (\text{original idx} + 4) \bmod 8.
$$

Here, the indices can be thought of as offsets within the group. To get the logical offset, or the `original idx`, we simply do

$$
x \bmod 8.
$$

Thus, the formula to get the physical offset, or `swapped idx`, is

$$
\text{physicalOffset} = ((x \bmod 8) + 4) \bmod 8
$$

Now that we have the physical offset within each group of 8 logical $x$ values, we need to get the physical $x$ coordinate $x'$. This is fairly easy to reason about. We already have the physical offset within the group, so we simply need to add it to the starting physical $x$ coordinate of that group of 8.

The group index is

$$
\text{groupIdx} = \left\lfloor \dfrac{x}{8} \right\rfloor,
$$

so the starting $x$ coordinate of the group is

$$
8 \cdot \text{groupIdx} = 8\left\lfloor \dfrac{x}{8} \right\rfloor.
$$

Thus, when we `swap` a group of 8 logical $x$ values, the physical coordinate $x'$ is

$$
x' = 8\left\lfloor \dfrac{x}{8} \right\rfloor + ((x \bmod 8) + 4) \bmod 8
$$

Thus our logical-to-physical mapping function is given by:

$$
F: L \rightarrow P
$$

where

```math
F(x,y) =
\begin{cases}
\left(
    8\left\lfloor \dfrac{x}{8} \right\rfloor + ((x \bmod 8) + 4) \bmod 8,y
\right)
&\text{ if } \left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4 \\[10pt]
(x,y)
&\text{ if } \left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 < 4.
\end{cases}
```

Here,

$$
L = P = \{0, 1, \dots, 127\} \times \{0, 1, \dots, 31\}
$$

where $L$ is the logical coordinate space and $P$ is the physical coordinate space.

First, we show that $F$ is well-defined. That is, we need to show that every $(x,y) \in L$ is mapped to exactly one element of $P$.

Let $(x,y)\in L$. Since $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8$ is an integer, exactly one of $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 < 4$ or $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4$ must hold. Therefore, exactly one case of $F$ applies.

If $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 < 4$, then $F(x,y)=(x,y)$. Since $(x,y)\in L$ and $L=P$, we immediately have $F(x,y)\in P$.

Now suppose $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4$. Let $g=\left\lfloor\dfrac{x}{8}\right\rfloor$ and $s=((x\bmod8)+4)\bmod8$. Then $F(x,y)=(8g+s,y)$. Since $(x,y)\in L$, we have $0\leq x\leq127$. Therefore, $0\leq g=\left\lfloor\dfrac{x}{8}\right\rfloor\leq15$.

Also, since $(x\bmod8)+4\geq0$ and $s=((x\bmod8)+4)\bmod8$, by definition of modulo 8, $0\leq s\leq7$. Thus,
$0\leq 8g+s\leq 8(15)+7=127$.

Since $y$ is unchanged and $(x,y)\in L$, we also have $0\leq y\leq31$. Therefore,

$$
F(x,y)=(8g+s,y)\in P.
$$

So in either case, $F$ maps every element of $L$ to exactly one element of $P$. Hence, $F$ is well-defined as a function

$$
F:L\rightarrow P.
$$

Now we prove that $F$ is a bijection. Since $F$ is piecewise, proving that it is one-to-one directly would require considering several separate cases. Instead, we can prove that $F$ has an inverse

$$
F^{-1}: P \rightarrow L.
$$

We can do this since $F$ is a bijection if and only if $F^{-1}$ exists.

First, define the candidate inverse function

$$
G: P \rightarrow L
$$

where

```math
G(x,y) =
\begin{cases}
\left(
    8\left\lfloor \dfrac{x}{8} \right\rfloor + ((x \bmod 8) + 4) \bmod 8,y
\right)
&\text{ if } \left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4 \\[10pt]
(x,y)
&\text{ if } \left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 < 4.
\end{cases}
```

Since $L=P$ and $G$ is defined by the same piecewise rule as $F$, the argument above showing that $F$ is well-defined also shows that $G$ is well-defined. Therefore,

$$
G:P\rightarrow L
$$

is a well-defined function.

To prove $F$ is invertible with inverse $G$, we must show that $\forall (x,y) \in L, (G \circ F)(x,y) = (x,y)$ and that $\forall (u,v) \in P, (F \circ G)(u,v) = (u,v)$.

First, we show that $\forall (x,y) \in L, (G \circ F)(x,y) = (x,y)$.

Let $(x,y) \in L$. Suppose $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 < 4$. Then

$$
(G \circ F)(x,y) = G(F(x,y)) = G(x,y) = (x,y)
$$

Now, suppose $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4$. Then

```math
(G \circ F)(x,y) = G(F(x,y))
=
G
\left(
    8\left\lfloor \dfrac{x}{8} \right\rfloor + ((x \bmod 8) + 4) \bmod 8,y
\right)
```

Now, let $g=\left\lfloor\dfrac{x}{8}\right\rfloor \in \mathbb{Z}$ and $r=x\bmod8 \in \mathbb{Z}, 0 \leq r < 8$. By the Division Algorithm,

$$
x = 8g + r
$$

Since $\left\lfloor \dfrac{x}{8} \right\rfloor \bmod 8 \geq 4$, we have $g \bmod 8 \geq 4$.

Now, let $s=(r+4)\bmod8$. Then $s\in\mathbb{Z}$ and $0\leq s<8$. Therefore,

$$
G
\left(
    8\left\lfloor \dfrac{x}{8} \right\rfloor + ((x \bmod 8) + 4) \bmod 8,y
\right) =
G(8g+s, y)
$$

Since $0 \leq s < 8$, we have

$$
\left\lfloor \dfrac{8g+s}{8} \right\rfloor = g.
$$

Therefore,

```math
\left\lfloor\dfrac{8g+s}{8}\right\rfloor\bmod8
=
g\bmod8
\geq4.
```

Also, since $((8g + s) \bmod8 + 4) \bmod 8 = (s + 4) \bmod 8$, we have

```math
G(8g+s,y)
=
\left(
8g+((s+4)\bmod8),
y
\right).
```

Since

$$
s=(r+4)\bmod8,
$$

we get

```math
\left(
8g+((s+4)\bmod8),
y
\right)
=
\left(
8g+
\left(
((r+4)\bmod8)+4
\right)\bmod8,
y
\right).
```

Using the property

$$
((a\bmod n)+b)\bmod n=(a+b)\bmod n \quad \text{for } a, b \in \mathbb{Z},
$$

we have

$$
\begin{aligned}
\left(((r+4)\bmod8)+4\right)\bmod8
&=(r+8)\bmod8\\
&=r,
\end{aligned}
$$

since $0\leq r<8$.

Therefore,

```math
\left(
8g+
\left(
((r+4)\bmod8)+4
\right)\bmod8,
y
\right)
=
(8g+r,y).
```

Earlier we showed that

$$
x=8g+r.
$$

Thus,

$$
(G\circ F)(x,y)=(x,y).
$$

Now, we need to show that $\forall (u,v)\in P,\quad (F\circ G)(u,v)=(u,v)$. However, notice that $L=P$ and $F$ and $G$ are defined by the exact same function. Therefore, $F=G$.

Thus,
```math
F\circ G
=
F\circ F
=
G\circ F.
```

We have already proved that $\forall (x,y)\in L$, $(G\circ F)(x,y)=(x,y)$.

Since $L=P$, this result also holds for every $(u,v)\in P$. Therefore,

```math
(F\circ G)(u,v)
=
(G\circ F)(u,v)
=
(u,v).
```

Thus,

$$
F\circ G=I_P
$$

and

$$
G\circ F=I_L.
$$

Therefore, $G=F^{-1}$, so $F$ is invertible. Hence, $F$ is a bijection.

### The Kernel

Since $F$ is a bijection, we can safely rearrange the physical layout of `bTile` according to this mapping without losing or overwriting any values. The mapping also gives us the desired bank-access pattern for both `float4` reads from `bTile`, removing the bank conflicts.

With both the `aTile` and `bTile` logical-to-physical mappings complete, we can now apply them to the kernel. The final kernel is shown below:

```cuda
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_X * BLOCK_Y;

constexpr uint32_t ROWS_PER_THREAD = 8;
constexpr uint32_t COLS_PER_THREAD = 8;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 32;

constexpr uint32_t FLOATS_PER_FLOAT4 = 4;
constexpr uint32_t FLOATS_PER_LOAD_PASS = NUM_THREADS * FLOATS_PER_FLOAT4;

constexpr uint32_t A_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_N;

constexpr uint32_t BYTES_PER_FLOAT4 = 16;

constexpr uint32_t FLOAT4_OFFSET_MASK = FLOATS_PER_FLOAT4 - 1u;
constexpr uint32_t TILE_M_WRAP_MASK = TILE_M - 1u;

constexpr uint32_t COLS_PER_THREAD_SHIFT = 3;


struct TileCoord{
    uint32_t x;
    uint32_t y;
};


__device__ __forceinline__ TileCoord aTilePhysicalCoord(uint32_t logicalX, uint32_t logicalY) {
    const uint32_t shift = logicalX & ~FLOAT4_OFFSET_MASK;
    return TileCoord{(logicalY + shift) & TILE_M_WRAP_MASK, logicalX};
}


__device__ __forceinline__ TileCoord bTilePhysicalCoord(uint32_t logicalX, uint32_t logicalY) {
    const uint32_t halfSwapMask = (logicalX >> COLS_PER_THREAD_SHIFT) & FLOATS_PER_FLOAT4;
    return TileCoord{logicalX ^ halfSwapMask, logicalY};
}


__global__ void matMulCudaResolveBankConflictsKernel (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ __align__(BYTES_PER_FLOAT4) float aTile[TILE_K][TILE_M];
    __shared__ __align__(BYTES_PER_FLOAT4) float bTile[TILE_K][TILE_N];

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tid = ty * BLOCK_X + tx;
    const uint32_t threadStartIdx = tid * FLOATS_PER_FLOAT4;

    const uint32_t aTileYStart = threadStartIdx / TILE_K;
    const uint32_t bTileYStart = threadStartIdx / TILE_N;

    const uint32_t aTileX = threadStartIdx % TILE_K;
    const uint32_t bTileX = threadStartIdx % TILE_N;

    const uint32_t ayBlockStart = blockIdx.y * TILE_M;
    const uint32_t bxBlockStart = blockIdx.x * TILE_N;

    const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
    const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;

    const uint32_t bTilePhysicalX = bTilePhysicalCoord(bTileX, 0).x;
    const uint32_t bTileReadXJ0 = bTilePhysicalCoord(tx * COLS_PER_THREAD + 0, 0).x;
    const uint32_t bTileReadXJ4 = bTilePhysicalCoord(tx * COLS_PER_THREAD + FLOATS_PER_FLOAT4, 0).x;

    __align__(BYTES_PER_FLOAT4) float aReg[ROWS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float bReg[COLS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            const uint32_t aTileY = tileRowOffset + aTileYStart;
            const TileCoord aTileCoords = aTilePhysicalCoord(aTileX, aTileY);
            const uint32_t ay = ayBlockStart + aTileY;;
            const uint32_t ax = tileStart + aTileX;

            const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
            aTile[aTileCoords.y + 0][aTileCoords.x] = toLoad.x;
            aTile[aTileCoords.y + 1][aTileCoords.x] = toLoad.y;
            aTile[aTileCoords.y + 2][aTileCoords.x] = toLoad.z;
            aTile[aTileCoords.y + 3][aTileCoords.x] = toLoad.w;
        }

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            const uint32_t bTileY = tileRowOffset + bTileYStart;
            const uint32_t by = tileStart + bTileY;
            const uint32_t bx = bxBlockStart + bTileX;

            *reinterpret_cast<float4*>(&bTile[bTileY][bTilePhysicalX]) = *reinterpret_cast<const float4*>(&b[by * n + bx]);
        }

        __syncthreads();

        #pragma unroll 16
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i += FLOATS_PER_FLOAT4) {
                const TileCoord aTileCoords = aTilePhysicalCoord(dotIdx, ty * ROWS_PER_THREAD + i);
                *reinterpret_cast<float4*>(&aReg[i]) = *reinterpret_cast<float4*>(&aTile[aTileCoords.y][aTileCoords.x]);
            }

            *reinterpret_cast<float4*>(&bReg[0]) = *reinterpret_cast<float4*>(&bTile[dotIdx][bTileReadXJ0]);
            *reinterpret_cast<float4*>(&bReg[4]) = *reinterpret_cast<float4*>(&bTile[dotIdx][bTileReadXJ4]);

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        const uint32_t cy = cyStart + i;

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
            const uint32_t cx = cxStart + j;
            *reinterpret_cast<float4*>(&c[cy * n + cx]) = *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}
```

The logical-to-physical mappings derived above are implemented in the `aTilePhysicalCoord` and `bTilePhysicalCoord` helper functions. These functions convert the logical shared memory coordinates into the physical coordinates used to store and read the values from `aTile` and `bTile`.

This kernel is still an exact-fit kernel, so it only works when the matrix dimensions are exactly divisible by the tile dimensions. Since

$$
TILE\_M = 128,\qquad TILE\_N = 128,\qquad TILE\_K = 32,
$$

we require

$$
m \bmod 128 = 0,\qquad
n \bmod 128 = 0,\qquad
k \bmod 32 = 0.
$$

This ensures that every output tile and every $K$-dimension tile is complete, so no boundary checks or partial tile handling are required.

A general version of this kernel is also included in the repository for matrix dimensions that are not exact multiples of the tile dimensions.

With the shared memory bank conflicts removed, the next step is to explicitly control the shape of the output tile assigned to each warp, rather than letting that shape be determined implicitly by the thread block layout.

## Matrix Kernel Optimization 7: Warp tiling

### What is Warp Tiling?

In the previous kernel, each thread computes an `8 x 8` tile of the output matrix. However, we never explicitly assigned an output tile to each warp. Instead, the shape of the warp tile was determined implicitly by the arrangement of threads within the `16 x 16` thread block.

Since each thread block contains `16 x 16` threads and each warp contains 32 threads, each block contains

$$
\dfrac{16 \text{ threads} \times 16 \text{ threads}}{32 \text{ threads per warp}} = 8 \text{ warps}.
$$

CUDA linearizes the threads in a block with the `x` dimension varying fastest. Therefore, each warp spans all 16 values of `tx` and two consecutive values of `ty`. The warps within the thread block are arranged as follows:

```text
16 x 16 thread block

   tx: 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

ty: +----------------------------------------------------------------+
 0  |                                                                |
 1  |                            Warp 0                              |
    +----------------------------------------------------------------+
 2  |                                                                |
 3  |                            Warp 1                              |
    +----------------------------------------------------------------+
 4  |                                                                |
 5  |                            Warp 2                              |
    +----------------------------------------------------------------+
 6  |                                                                |
 7  |                            Warp 3                              |
    +----------------------------------------------------------------+
 8  |                                                                |
 9  |                            Warp 4                              |
    +----------------------------------------------------------------+
10  |                                                                |
11  |                            Warp 5                              |
    +----------------------------------------------------------------+
12  |                                                                |
13  |                            Warp 6                              |
    +----------------------------------------------------------------+
14  |                                                                |
15  |                            Warp 7                              |
    +----------------------------------------------------------------+
```

Each warp therefore covers a `2 x 16` region of threads within the block. Since each thread computes an `8 x 8` output tile, each warp computes an output tile of size

$$
(2 \cdot 8) \times (16 \cdot 8) = 16 \times 128.
$$

The eight warps are stacked vertically within the block, so together they cover

$$
(8 \cdot 16) \times (16 \cdot 8) = 128 \times 128,
$$

which matches the `128 x 128` output tile computed by the entire thread block.

Therefore, we already have warp tiling in the sense that each warp computes its own output tile. However, the shape and location of that tile are currently determined implicitly by the thread block layout. The warp tiling optimization makes this mapping explicit, allowing us to directly control the shape of the output tile assigned to each warp.

The reason explicitly controlling the warp tile can be useful is that threads in a warp execute instructions together. When the threads in a warp load values from shared memory, the addresses requested by those threads are therefore accessed together. Changing the shape of the output tile assigned to a warp changes which `aTile` and `bTile` values are requested together by the 32 threads.

For example, with the current `16 x 128` warp tile, the warp spans only two thread tiles in the $y$ direction but 16 thread tiles in the $x$ direction. During the accumulation phase, when each thread loads the `aTile` and `bTile` values needed for the current `dotIdx` into `aReg` and `bReg`, many threads within the warp request the same values from `aTile`, while they request many different values from `bTile`.

```text
Implicit warp tile: 16 x 128

Each Ti = T0, T1, ..., T31 is one 8 x 8 thread tile.

  threadTileX: 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
threadTileY: +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
          0  | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |T10 |T11 |T12 |T13 |T14 |T15 |
             +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
          1  |T16 |T17 |T18 |T19 |T20 |T21 |T22 |T23 |T24 |T25 |T26 |T27 |T28 |T29 |T30 |T31 |
             +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+

Warp tile = 2 thread tiles high x 16 thread tiles wide
          = (2 x 8) x (16 x 8)
          = 16 x 128 outputs

A access pattern:
- only 2 thread tile rows
- for a given A register load, many threads request the same aTile values

B access pattern:
- 16 thread tile columns
- for a given B register load, threads request many different bTile values
```

If we instead make the warp tile taller and narrower, this relationship changes. During these same register loads, the warp requests more distinct values from `aTile`, but fewer distinct values from `bTile`.

For example, consider the following:

```text
Example explicit warp tile: 128 x 16

Each Ti = T0, T1, ..., T31 is one 8 x 8 thread tile.

  threadTileX: 0    1
threadTileY: +----+----+
          0  | T0 | T1 |
             +----+----+
          1  | T2 | T3 |
             +----+----+
          2  | T4 | T5 |
             +----+----+
          3  | T6 | T7 |
             +----+----+
          4  | T8 | T9 |
             +----+----+
          5  |T10 |T11 |
             +----+----+
          6  |T12 |T13 |
             +----+----+
          7  |T14 |T15 |
             +----+----+
          8  |T16 |T17 |
             +----+----+
          9  |T18 |T19 |
             +----+----+
         10  |T20 |T21 |
             +----+----+
         11  |T22 |T23 |
             +----+----+
         12  |T24 |T25 |
             +----+----+
         13  |T26 |T27 |
             +----+----+
         14  |T28 |T29 |
             +----+----+
         15  |T30 |T31 |
             +----+----+

Warp tile = 16 thread tiles high x 2 thread tiles wide
          = (16 x 8) x (2 x 8)
          = 128 x 16 outputs

A access pattern:
- 16 thread tile rows
- for a given A register load, threads request many different aTile values

B access pattern:
- only 2 thread tile columns
- for a given B register load, many threads request the same bTile values
```

Therefore, changing the warp tile does not change the amount of work performed by the warp or the number of outputs computed by each thread. Instead, it changes how the shared memory accesses of the 32 threads are grouped together during the accumulation phase. This can change shared memory broadcasts, bank conflicts, and the overall cost of supplying operands to the warp.

A shared memory broadcast occurs when multiple threads in a warp access the same shared memory address. The value only needs to be read from shared memory once and can then be broadcast to all threads in the warp that requested it.

There is not necessarily one warp shape that is always better. Instead, making the warp tile explicit gives us another parameter that we can experiment with and tune for the GPU.

### The Kernel

With warp tiling and its benefits explained, below is the explicitly warp tiled kernel, where each warp computes a `128 x 16` output tile instead of the implicit `16 x 128` output tile from the previous kernel:

```cuda
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t NUM_THREADS = BLOCK_X * BLOCK_Y;

constexpr uint32_t ROWS_PER_THREAD = 8;
constexpr uint32_t COLS_PER_THREAD = 8;

constexpr uint32_t TILE_M = BLOCK_Y * ROWS_PER_THREAD;
constexpr uint32_t TILE_N = BLOCK_X * COLS_PER_THREAD;
constexpr uint32_t TILE_K = 32;

constexpr uint32_t WARP_TILE_M = 128;
constexpr uint32_t WARP_TILE_N = 16;

constexpr uint32_t WARP_TILES_PER_BLOCK_N = TILE_N / WARP_TILE_N;
constexpr uint32_t THREAD_TILES_PER_WARP_N = WARP_TILE_N / COLS_PER_THREAD;

constexpr uint32_t FLOATS_PER_FLOAT4 = 4;
constexpr uint32_t FLOATS_PER_LOAD_PASS = NUM_THREADS * FLOATS_PER_FLOAT4;

constexpr uint32_t A_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_K;
constexpr uint32_t B_TILE_ROW_STRIDE = FLOATS_PER_LOAD_PASS / TILE_N;

constexpr uint32_t BYTES_PER_FLOAT4 = 16;

constexpr uint32_t FLOAT4_OFFSET_MASK = FLOATS_PER_FLOAT4 - 1u;
constexpr uint32_t TILE_M_WRAP_MASK = TILE_M - 1u;
constexpr uint32_t COLS_PER_THREAD_SHIFT = 3;

constexpr uint32_t THREADS_PER_WARP = 32;

struct TileCoord{
    uint32_t x;
    uint32_t y;
};


__device__ __forceinline__ TileCoord aTilePhysicalCoord(uint32_t logicalX, uint32_t logicalY) {
    const uint32_t shift = logicalX & ~FLOAT4_OFFSET_MASK;
    return TileCoord{(logicalY + shift) & TILE_M_WRAP_MASK, logicalX};
}


__device__ __forceinline__ TileCoord bTilePhysicalCoord(uint32_t logicalX, uint32_t logicalY) {
    const uint32_t halfSwapMask = (logicalX >> COLS_PER_THREAD_SHIFT) & FLOATS_PER_FLOAT4;
    return TileCoord{logicalX ^ halfSwapMask, logicalY};
}


__global__ void matMulCudaWarpTilingExactFitKernel (
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t m,
    uint32_t n,
    uint32_t k
) {
    __shared__ __align__(BYTES_PER_FLOAT4) float aTile[TILE_K][TILE_M];
    __shared__ __align__(BYTES_PER_FLOAT4) float bTile[TILE_K][TILE_N];

    const uint32_t ty = threadIdx.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tid = ty * BLOCK_X + tx;
    const uint32_t threadStartIdx = tid * FLOATS_PER_FLOAT4;

    const uint32_t warpId = tid / THREADS_PER_WARP;
    const uint32_t warpThreadId = tid % THREADS_PER_WARP;

    const uint32_t warpY = warpId / WARP_TILES_PER_BLOCK_N;
    const uint32_t warpX = warpId % WARP_TILES_PER_BLOCK_N;

    const uint32_t warpThreadY = warpThreadId / THREAD_TILES_PER_WARP_N;
    const uint32_t warpThreadX = warpThreadId % THREAD_TILES_PER_WARP_N;

    const uint32_t threadTileStartYInBlock = warpY * WARP_TILE_M + warpThreadY * ROWS_PER_THREAD;
    const uint32_t threadTileStartXInBlock = warpX * WARP_TILE_N + warpThreadX * COLS_PER_THREAD;

    const uint32_t aTileYStart = threadStartIdx / TILE_K;
    const uint32_t bTileYStart = threadStartIdx / TILE_N;

    const uint32_t aTileX = threadStartIdx % TILE_K;
    const uint32_t bTileX = threadStartIdx % TILE_N;

    const uint32_t ayBlockStart = blockIdx.y * TILE_M;
    const uint32_t bxBlockStart = blockIdx.x * TILE_N;

    const uint32_t cyStart = ayBlockStart + threadTileStartYInBlock;
    const uint32_t cxStart = bxBlockStart + threadTileStartXInBlock;

    const uint32_t bTilePhysicalX = bTilePhysicalCoord(bTileX, 0).x;

    __align__(BYTES_PER_FLOAT4) float aReg[ROWS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float bReg[COLS_PER_THREAD];
    __align__(BYTES_PER_FLOAT4) float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    for (uint32_t tileStart = 0; tileStart < k; tileStart += TILE_K) {

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_M; tileRowOffset += A_TILE_ROW_STRIDE) {
            const uint32_t aTileY = tileRowOffset + aTileYStart;
            const TileCoord aTileCoords = aTilePhysicalCoord(aTileX, aTileY);
            const uint32_t ay = ayBlockStart + aTileY;
            const uint32_t ax = tileStart + aTileX;

            const float4 toLoad = *reinterpret_cast<const float4*>(&a[ay * k + ax]);
            aTile[aTileCoords.y + 0][aTileCoords.x] = toLoad.x;
            aTile[aTileCoords.y + 1][aTileCoords.x] = toLoad.y;
            aTile[aTileCoords.y + 2][aTileCoords.x] = toLoad.z;
            aTile[aTileCoords.y + 3][aTileCoords.x] = toLoad.w;
        }

        #pragma unroll
        for (uint32_t tileRowOffset = 0; tileRowOffset < TILE_K; tileRowOffset += B_TILE_ROW_STRIDE) {
            const uint32_t bTileY = tileRowOffset + bTileYStart;
            const uint32_t by = tileStart + bTileY;
            const uint32_t bx = bxBlockStart + bTileX;

            *reinterpret_cast<float4*>(&bTile[bTileY][bTilePhysicalX]) = *reinterpret_cast<const float4*>(&b[by * n + bx]);
        }

        __syncthreads();

        #pragma unroll 16
        for (uint32_t dotIdx = 0; dotIdx < TILE_K; dotIdx++) {

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i += FLOATS_PER_FLOAT4) {
                const TileCoord aTileCoords = aTilePhysicalCoord(dotIdx, threadTileStartYInBlock + i);
                *reinterpret_cast<float4*>(&aReg[i]) = *reinterpret_cast<float4*>(&aTile[aTileCoords.y][aTileCoords.x]);
            }

            #pragma unroll
            for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
                const TileCoord bTileCoords = bTilePhysicalCoord(threadTileStartXInBlock + j, dotIdx);
                *reinterpret_cast<float4*>(&bReg[j]) = *reinterpret_cast<float4*>(&bTile[bTileCoords.y][bTileCoords.x]);
            }

            #pragma unroll
            for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
                #pragma unroll
                for (uint32_t j = 0; j < COLS_PER_THREAD; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < ROWS_PER_THREAD; i++) {
        const uint32_t cy = cyStart + i;

        #pragma unroll
        for (uint32_t j = 0; j < COLS_PER_THREAD; j += FLOATS_PER_FLOAT4) {
            const uint32_t cx = cxStart + j;
            *reinterpret_cast<float4*>(&c[cy * n + cx]) = *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}
```

`WARP_TILE_M = 128` and `WARP_TILE_N = 16` control the size of the output tile computed by each warp. Since each thread still computes an `8 x 8` output tile, a `128 x 16` warp tile contains

$$
\dfrac{128 \text{ output rows per warp tile}}
{8 \text{ output rows per thread tile row}}
\times
\dfrac{16 \text{ output columns per warp tile}}
{8 \text{ output columns per thread tile column}}=
16 \text{ thread tile rows}
\times
2 \text{ thread tile columns}.
$$

Since a warp contains 32 threads and each thread computes one thread tile, this gives exactly one `8 x 8` thread tile per thread.

You'll also see that `warpId` and `warpThreadId` have been calculated. `warpId` is the ID of the warp within the block. There are

$$
\dfrac{256 \text{ threads per block}}{32 \text{ threads per warp}} = 8 \text{ warps per block}
$$

Thus, $\text{warpId} \in \{0, 1, \dots, 7\}$, and it's calculated by

$$
\text{warpId} = \left\lfloor \dfrac{\text{tid}}{32} \right\rfloor
$$

since each consecutive group of 32 thread IDs belongs to one warp.

`warpThreadId` is the ID of a thread within its warp and is given by

$$
\text{warpThreadId} = \text{tid}\bmod32
$$

Thus, $\text{warpThreadId} \in \{0, 1, \dots, 31\}$.

Next, `warpX` and `warpY` give the coordinates of the warp tile within the block's `128 x 128` output tile. Each block computes

```text
TILE_N = BLOCK_X x COLS_PER_THREAD = 16 x 8 = 128
```

output columns, while each warp tile computes `WARP_TILE_N = 16` output columns. Therefore, the block contains

$$
\dfrac{128 \text{ output columns per block tile}}{16 \text{ output columns per warp tile}} = 8 \text{ warp tile columns per block tile}
$$

Since `warpId` identifies one of the eight warps in the block, its warp tile coordinates are

$$
\text{warpY} =
\left\lfloor
\dfrac{\text{warpId}}{8}
\right\rfloor
$$

and

$$
\text{warpX} =
\text{warpId}\bmod8.
$$

Because $\text{warpId} \in \{0, 1, \dots, 7\}$, we have $\text{warpY} = 0$ and $\text{warpX} \in \{0, 1, \dots, 7\}$.

So each `128 x 128` block output tile is divided into eight `128 x 16` warp tiles:

```text
Each Wi = W0, W1, ..., W7 is one 128 x 16 warp tile.

 warpX:  0    1    2    3    4    5    6    7
warpY: +----+----+----+----+----+----+----+----+
   0   | W0 | W1 | W2 | W3 | W4 | W5 | W6 | W7 |
       +----+----+----+----+----+----+----+----+
```

Next, `warpThreadY` and `warpThreadX` give the coordinates of a thread tile within its warp tile. Each warp tile contains `WARP_TILE_N = 16` output columns, while each thread tile contains `COLS_PER_THREAD = 8` output columns. Therefore, there are

$$
\dfrac{16 \text{ output columns per warp tile}}
{8 \text{ output columns per thread tile}} =
2 \text{ thread tile columns per warp tile}.
$$

Since `warpThreadId` is the linear ID of the thread within the warp, we can convert it into thread tile coordinates using

$$
\text{warpThreadY}
= \left\lfloor
\dfrac{\text{warpThreadId}}{2}
\right\rfloor
$$

and

$$
\text{warpThreadX} =
\text{warpThreadId}\bmod2.
$$

Thus, $\text{warpThreadY} \in \{0, 1, \dots, 15\}$ as $\text{warpThreadId} \in \{0, 1, \dots, 31\}$ and $\text{warpThreadX} \in \{0, 1\}$.

Each `128 x 16` warp tile therefore looks like:

```text
Each Ti = T0, T1, ..., T31 is one 8 x 8 thread tile.

 warpThreadX:  0    1
warpThreadY: +----+----+
          0  | T0 | T1 |
             +----+----+
          1  | T2 | T3 |
             +----+----+
          2  | T4 | T5 |
             +----+----+
          3  | T6 | T7 |
             +----+----+
          4  | T8 | T9 |
             +----+----+
          5  |T10 |T11 |
             +----+----+
          6  |T12 |T13 |
             +----+----+
          7  |T14 |T15 |
             +----+----+
          8  |T16 |T17 |
             +----+----+
          9  |T18 |T19 |
             +----+----+
         10  |T20 |T21 |
             +----+----+
         11  |T22 |T23 |
             +----+----+
         12  |T24 |T25 |
             +----+----+
         13  |T26 |T27 |
             +----+----+
         14  |T28 |T29 |
             +----+----+
         15  |T30 |T31 |
             +----+----+
```

The final indexing step is to calculate `threadTileStartYInBlock` and `threadTileStartXInBlock`. These are the starting coordinates of the thread's `8 x 8` output tile relative to the block's `128 x 128` output tile.

For the $y$ coordinate, we first calculate the $y$ coordinate, relative to the block output tile, at which the warp's first thread tile begins.

```text
warpStartY = warpY * WARP_TILE_M
```

For this particular `128 x 16` warp layout, `warpY = 0` and `WARP_TILE_M = 128` so

$$
\text{warpStartY}= 0\cdot128= 0.
$$

Within the warp tile, the thread tile begins at

```text
threadStartYInWarp = warpThreadY * ROWS_PER_THREAD.
```

Thus, to get the starting $y$ coordinate of a particular thread tile relative to the entire block output tile, we add the $y$ coordinate of the first thread tile in the warp, `warpStartY`, to the starting $y$ coordinate of the current thread tile relative to that warp, `threadStartYInWarp`:

```text
threadTileStartYInBlock = warpStartY + threadStartYInWarp,
```

or directly,

```text
threadTileStartYInBlock = warpY * WARP_TILE_M + warpThreadY * ROWS_PER_THREAD.
```

The same logic applies for calculating `threadTileStartXInBlock`:

```text
threadTileStartXInBlock = warpX * WARP_TILE_N + warpThreadX * COLS_PER_THREAD.
```

The next change is to `cyStart` and `cxStart`. Logically, these calculations still serve the same purpose as in the previous kernel: they determine the global starting coordinates of the thread's output tile in `C`. The difference is that the thread's position inside the block output tile is now determined by the explicit warp mapping rather than directly by `ty` and `tx`.

Therefore,

```cuda
const uint32_t cyStart = ayBlockStart + threadTileStartYInBlock;
const uint32_t cxStart = bxBlockStart + threadTileStartXInBlock;
```

replaces the previous mapping

```cuda
const uint32_t cyStart = ayBlockStart + ty * ROWS_PER_THREAD;
const uint32_t cxStart = bxBlockStart + tx * COLS_PER_THREAD;
```

Lastly, in the accumulator loops, you'll see we have replaced `ty * ROWS_PER_THREAD` and `tx * COLS_PER_THREAD` with `threadTileStartYInBlock` and `threadTileStartXInBlock` respectively. This makes sense because in the implicitly warp tiled kernel `ty * ROWS_PER_THREAD` is the thread tile $y$ start coordinate and `tx * COLS_PER_THREAD` is the thread tile $x$ start coordinate.

That is the main change introduced by explicit warp tiling. The amount of work performed by the block, warp, and individual thread has not changed. The block still computes a `128 x 128` output tile, each warp still computes 2048 output values, and each thread still computes an `8 x 8` output tile. What has changed is how those thread tiles are grouped into warps and where each warp is placed within the block output tile.

After benchmarking, the `128 x 16` warp tiled kernel was about 6% faster than the previous kernel. Interestingly, profiling showed that this new warp layout reintroduced shared memory bank conflicts.

Despite these additional conflicts, the kernel was still faster overall. This is because changing the warp shape also changes how the 32 threads share and access `aTile` and `bTile`, as well as the overall execution pattern of the warp. The cost of the additional bank conflicts was therefore outweighed by the benefits of the new warp mapping.

This is an important result: minimizing bank conflicts does not necessarily minimize the total execution time of the kernel. Warp shape introduces several interacting tradeoffs, so the fastest configuration cannot always be determined by optimizing a single metric in isolation.

Since the warp tile shape is now an explicit parameter, the next step is to autotune the kernel by testing different combinations of block tile, warp tile, thread tile, and `TILE_K` sizes to determine which configuration performs best on the GPU.
