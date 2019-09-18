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

    const size_t offs_sA = bi * si * K;
    const size_t offs_sB = bj * sj;

    // Copy to local memories
    for (size_t i = 0; i < si; i++) {
        for (size_t k = 0; k < K; k++) {
            sA[i * K + k] = A[offs_sA + i * K + k];
        }
    }
    for (size_t j = 0; j < sj; j++) {
        for (size_t k = 0; k < K; k++) {
            sB[k * sj + j] = B[offs_sB + k * N + j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate (li, lj) component of the work group
    float res = 0;
    for (size_t k = 0; k < K; k++) {
        res += sA[li * K + k] * sB[k * sj + lj];
    }

    const size_t gi = bi * si + li; // get_global_id(0)
    const size_t gj = bj * sj + lj; // get_global_id(1)
    C[gi * N + gj] = res;
}
