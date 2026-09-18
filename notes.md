# CUDA Matrix Optimization

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

A memory instruction happens at the warp level and tells threads in a warp to load or store data. A memory transaction is a logical memory system operation used to service an instruction given to a warp. One instruction can require multiple transactions, depending on the addresses requested by the threads. Thus, the sequence is.

1. Warp executes a memory instruction
2. Each active thread/lane computes the address it needs
3. Those per thread memory requests are collected
4. The memory subsystem examines the requested addresses.
5. It generates the required memory transactions
6. Those transactions service the warp’s requests

For newer NVIDIA GPUs, a warp's global memory accesses are serviced using 32 byte aligned memory segments - aligned meaning each segment begins at an address that is a multiple of 32 bytes. For example:

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

Since `a[0]` through `a[7]` occupy exactly 32 contiguous bytes and `a[0]` begins on a 32 byte aligned address, all 8 requests fit within **one** 32 byte memory segment. Therefore, all 8 memory requests are serviced by **one** memory transaction. Thus, we have perfect coalescing.

Overall, memory coalescing reduces the number of memory transactions required to service a warp's memory instruction and can therefore improve kernel performance compared with a poorly coalesced access pattern.

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

## Matrix Optimization 4: 2D Register Tiling

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

## Matrix Optimization 5: Vectorization

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

This can easily be done using the same general loading logic we used in the [2D register tiled kernel](#matrix-optimization-4-2d-register-tiling). We have 32 threads in our warp, so we increment `i` by 32 so that each iteration skips over the 32 reads already performed by the previous iteration.

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

