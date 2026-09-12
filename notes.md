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

For example, a `16 × 16` block contains 256 threads, forming 8 `(256 / 32 = 8)` warps. Its first warp contains:

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
3. Those per-thread memory requests are collected
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

The expression `a[i]` corresponds to a **single** memory load instruction issued for the warp. Each of the 8 active threads executes that same instruction and generates one 4-byte memory request:

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

Assume `k = 256`, `n = 1024`, with `16 × 16` = `256` threads per block. Also assume the arrays begin at addresses aligned to 32 bytes and all threads in the warp we examine are within the matrix bounds.

We will examine the first warp of block `(0, 0)` at loop iteration `i = 0`. Denote a thread as $t_i$, $i \in \{0, 1, \dots, 31\}$. Their (`threadIdx.x`, `threadIdx.y`) coordinates are:

- $t_0: (0, 0), t_1: (1, 0), \dots, t_{15}: (15, 0)$
- $t_{16}: (0, 1), t_{17}: (1, 1), \dots, t_{31}: (15, 1)$

First, consider the load from `a`: `a[y * k + i]`

At this iteration, threads 0–15 have `y = 0`, so they all read `a[0]`. Threads 16–31 have `y = 1`, so they all read `a[256]`. These two values fall in different 32-byte segments:

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

The 32 threads read 16 different contiguous values from `b`, indices 0-15. These occupy two consecutive 32-byte segments:

```text
Segment 0:  b[0] through b[7]  - threads 0-7, and 16-23
Segment 1: b[8] through b[15]  - threads 8-15, and 24-31
```

The hardware combines the requests from threads 0-7, and 16-23 into one memory transaction and those from threads 8-15, and 24-31 into another. Thus, two memory transactions are needed to load the values requested by the warp.

Here, both memory transactions are fully utilized because every float in each segment is used by the warp. The 16 distinct floats occupy 64 bytes, so two 32-byte transactions are the minimum needed to load them.

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

The 32 threads write to 32 different addresses of `c`. These occupy four 32-byte segments:

```text
Segment 0:         c[0] through c[7]  - threads 0-7
Segment 1:        c[8] through c[15]  - threads 8-15
Segment 128: c[1024] through c[1031]  - threads 16-23
Segment 129: c[1032] through c[1039]  - threads 24-31
```

The hardware combines the write requests from threads 0-7 into one memory transaction, threads 8-15 into another, threads 16-23 into another, and threads 24-31 into another. Thus, four memory transactions are needed to write the warp's results.

Here, all four memory transactions are fully utilized because the warp writes to every float in each segment. The 32 distinct floats occupy 128 bytes, so four 32-byte transactions are the minimum needed to write them.

Here is another example of perfect coalescing.

Our naive kernel already coalesces its global-memory accesses. `b` reads and `c` writes fully utilize the accessed segments under the conditions analyzed above, while `a` reads combine repeated requests but use only a small portion of each segment per iteration.
