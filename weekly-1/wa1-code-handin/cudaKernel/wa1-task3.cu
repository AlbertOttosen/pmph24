#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300

__global__ void myKernel(float* X, float *Y, int N) {
    // Calculate global thread index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if gid is within array bounds
    if (gid < N) {
        double temp = X[gid] / (X[gid] - 2.3);
        Y[gid] = temp * temp * temp; // Efficient cube calculation
    }
}

void sequential(float* X, float* Y, int N) {
    for (int i = 0; i < N; ++i) {
        double temp = X[i] / (X[i] - 2.3);
        Y[i] = temp * temp * temp;
    }
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
        if (argc == 1) {
            N = 753411;
        }
        else if (argc == 2) { 
            N = atoi(argv[1]);
        }
        else {   
            printf("Num Args is: %d instead of 0 or 1. Exiting!\n", argc); 
            exit(1);
        }

        printf("N is: %d\n", N);

        // const unsigned int maxN = 500000000;
        // if(N > maxN) {
        //     printf("N is too big; maximal value is %d. Exiting!\n", maxN);
        //     exit(2);
        // }
    }

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);

    // Set a fixed seed for reproducibility
    srand(42);

    // Fill the array with random float values between 1 and 10
    for (unsigned int i = 0; i < N; ++i) {
        h_in[i] = 1.0f + (float)rand() / (float)(RAND_MAX / 9.0f);  // Scale to range [1, 10]
    }

    float* h_out = (float*) malloc(mem_size);

    // allocate memory for sequential result
    float* seq_out = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)i;
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    unsigned int B = 256;
    unsigned int numblocks = (N + B - 1) / B; // number of blocks in dimension x
    dim3 block(B, 1, 1), grid(numblocks, 1, 1); // total number of threads (numblocks*B) may overshoot N!

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        myKernel<<< grid, block>>>(d_in, d_out, N);
    }
  
    double elapsed_gpu; 
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            myKernel<<< grid, block>>>(d_in, d_out, N);
        }
        cudaDeviceSynchronize();
        // ^ `cudaDeviceSynchronize` is needed for runtime
        //     measurements, since CUDA kernels are executed
        //     asynchronously, i.e., the CPU does not wait
        //     for the kernel to finish.
        //   However, `cudaDeviceSynchronize` is expensive
        //     so we need to amortize it across many runs;
        //     hence, when measuring performance use a big
        //     N and increase GPU_RUNS to 100 or more.
        //   Sure, it would be better by using CUDA events, but
        //     the current procedure is simple & works well enough.
        //   Please note that the execution of multiple
        //     kernels in Cuda executes correctly without such
        //     explicit synchronization; we need this only for
        //     runtime measurement.
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_gpu = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed_gpu * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", elapsed_gpu, gigabytespersec);
        
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    double elapsed_cpu; 
    { // run sequential implementation
      // just a single run since compiler optimizes redundant work
        
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        sequential(h_in, seq_out, N);
        
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed_cpu = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec));
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed_cpu * 1000.0);
        printf("The cpu took on average %f microseconds. GB/sec: %f \n", elapsed_cpu, gigabytespersec);
    }

    double speedup = elapsed_cpu / elapsed_gpu;

    // print result
    //for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);
    printf("Speedup = %f microseconds / %f microseconds = %f \n", elapsed_cpu, elapsed_gpu, speedup);

    for(unsigned int i=0; i<N; ++i) {
        float actual   = h_out[i];
        float expected = pow(h_in[i] / (h_in[i] - 2.3), 3); 
        if( actual != expected ) {
            printf("INVALID result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            exit(3);
        }
    }
    printf("VALID\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}