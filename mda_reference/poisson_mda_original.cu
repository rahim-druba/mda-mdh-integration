#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 1024
#define h (1.0 / (N - 1.0))
#define MAX_ITER 10000000 // Количество итераций для метода Якоби

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void iteratePoisson(double* u_old, const double* f, double* u_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;
 bool inBounds = (i >= 1 && i < N - 1 && j >= 1 && j < N - 1);
 if (inBounds) {
 double top = u_old[(j - 1) * N + i];
 double bot = u_old[(j + 1) * N + i];
 double right = u_old[j * N + (i + 1)];
 double left = u_old[j * N + (i - 1)];
 int n = j * N + i;
 double newVal = 0.25 * (top + bot + left + right + f[n] * h * h);
 u_new[j * N + i] = newVal;
 }

}


int main() {
    printf("Starting 2D Poisson solver (True Jacobi via MDA)...\n");
    size_t size = N * N * sizeof(double);

    double* h_u_old = (double*)malloc(size);
    double* d_u_old = NULL;
    double* h_f = (double*)malloc(size);
    double* d_f = NULL;
    double* h_u_new = (double*)malloc(size);
    double* d_u_new = NULL;


    for (int i = 0; i < N * N; i++) {
        h_u_old[i] = 0.0; 
        h_f[i] = 1.0; 
        h_u_new[i] = 0.0; 
    }

    printf("Executing CUDA Steps...\n");
    
    CHECK_CUDA(cudaMalloc((void**)&d_u_old, size));
    CHECK_CUDA(cudaMemcpy(d_u_old, h_u_old, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_f, size));
    CHECK_CUDA(cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_u_new, size));
    CHECK_CUDA(cudaMemcpy(d_u_new, h_u_new, size, cudaMemcpyHostToDevice));
    

    dim3 block(32, 32);
    dim3 grid((N + 32 - 1) / 32, (N + 32 - 1) / 32);
    
    printf("Launching Kernel: iteratePoisson for %d iterations...\n", MAX_ITER);
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        iteratePoisson<<<grid, block>>>(d_u_old, d_f, d_u_new);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        double* tmp = d_u_old; 
        d_u_old = d_u_new; 
        d_u_new = tmp;
    }

    printf("Copying results back to CPU...\n");
    CHECK_CUDA(cudaMemcpy(h_u_old, d_u_old, size, cudaMemcpyDeviceToHost)); 

    printf("Cleaning up memory...\n");
    free(h_u_old);
    cudaFree(d_u_old);
    free(h_f);
    cudaFree(d_f);
    free(h_u_new);
    cudaFree(d_u_new);

    printf("Execution completed successfully.\n");
    return 0;
}
