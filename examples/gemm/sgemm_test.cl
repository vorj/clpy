// fetch(array name, offset, vertical id (dim 0), horizontal id (dim 1), vertical size)
#define fetch(arr, offs, i, j, m) arr[offs + j * m + i]

__kernel void sgemm(
        const ulong M,
        const ulong N,
        const ulong K,
        __global const float* A,
        CArray_2 const info_A,
        __global const float* B,
        CArray_2 const info_B,
        __global float* C,
        CArray_2 const info_C,
        __local float* sA,
        __local float* sB
) {
    const size_t li = get_local_id(0);
    const size_t lj = get_local_id(1);

    const size_t si = get_local_size(0);
    const size_t sj = get_local_size(1);

    const size_t bi = get_group_id(0);
    const size_t bj = get_group_id(1);

    // const size_t offs_sA = bi * si * K;
    const size_t offs_sA = bi * si;
    // const size_t offs_sB = bj * sj;
    const size_t offs_sB = bj * sj * K;

    // Copy to local memories
    for (size_t i = 0; i < si; i++) {
        for (size_t k = 0; k < K; k++) {
            fetch(sA, 0, i, k, si) = fetch(A, offs_sA, i, k, M);
        }
    }
    for (size_t j = 0; j < sj; j++) {
        for (size_t k = 0; k < K; k++) {
            fetch(sB, 0, k, j, K) = fetch(B, offs_sB, k, j, K);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate (li, lj) component of the work group
    float res = 0;
    for (size_t k = 0; k < K; k++) {
        res += fetch(sA, 0, li, k, si) * fetch(sB, 0, k, lj, M);
    }

    const size_t gi = bi * si + li; // get_global_id(0)
    const size_t gj = bj * sj + lj; // get_global_id(1)
    fetch(C, 0, gi, gj, M) = res;
}
