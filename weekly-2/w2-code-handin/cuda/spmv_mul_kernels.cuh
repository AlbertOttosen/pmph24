#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    // Calculate global thread index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if gid is within array bounds
    if (gid >= tot_size) return;

    // Fill array with 0
    flags_d[idx] = 0;
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    // Get the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if gid is within array bounds
    if (idx >= mat_rows) return;

    // Fill array with 0
    flags_d[idx] = 0;

    // Set the flags where mat_shp_sc_d is non-negative
    if (mat_shp_sc_d[idx] != -1) {
        flags_d[mat_shp_sc_d[idx]] = 1;
    }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    // Calculate global thread index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if gid is within array bounds
    if (gid >= tot_size) return;

    //let mul_mat = map (\(i,x) -> x*vct[i]) mat_val
    tmp_pairs[gid] = mat_vals[gid] * vct[mat_inds[gid]];
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    // Calculate global thread index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if gid is within array bounds
    if (gid >= mat_rows) return;
    
    // Get index of last value in segment
    int end_of_segment = mat_shp_sc_d[gid];

    // Get the result of the segmented scan at that index
    res_vct_d[gid] = tmp_scan[end_of_segment];
}

#endif // SPMV_MUL_KERNELS
