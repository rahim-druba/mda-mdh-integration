// GPU-only block-size sweep harness for MDA's naive kernel (mda_kernel.cu).
// Same caching approach as bench_gpu_sweep_mdh.cu: reads the serial baseline
// from bench/serial_cache_NX<N>.txt instead of recomputing it per block config.
// The only change from the original hardcoded dim3 block(32,32) in
// test_poisson_mda.cu is making the block shape overridable via -DBLOCK_X/-DBLOCK_Y.
//
// Build (example, 16x8 block at N=1024):
//   nvcc bench/bench_gpu_sweep_mda.cu bench/mda_kernel.cu \
//     -o /tmp/sweep_mda_1024_16x8 -DNX_GRID=1024 -DN=1024 -DBLOCK_X=8 -DBLOCK_Y=16
//
// Prints: CSVROW,mda,N,AxB,iterations,gpu_s,cpu_s,speedup,u_center_gpu,u_center_serial,gpu_exact_err

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#ifndef NX_GRID
#define NX_GRID 64
#endif
#ifndef BLOCK_X
#define BLOCK_X 32
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

static const int    NX          = NX_GRID;
static const int    NI          = NX - 2;
static const double H           = 1.0 / (NX - 1.0);
static const double H2          = H * H;
static const double EPS         = 1e-4;
static const int    N_ITER_MAX  = 3000000;

extern __global__ void iteratePoisson(double* u_old, const double* f, double* u_new);

#define CUDA_CHECK(e) do {                                              \
    cudaError_t _err = (e);                                            \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr,"CUDA error '%s' at %s:%d\n",                  \
                cudaGetErrorString(_err),__FILE__,__LINE__);           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

static inline double source_fn(double x, double y) {
    return 2.0 * (x*(1.0-x) + y*(1.0-y));
}

static double compute_residual(const double *u, const double *src, int ni, double h2) {
    double max_res = 0.0;
    for (int k = 0; k < ni; k++)
        for (int l = 0; l < ni; l++) {
            double above  = (k > 0)    ? u[(k-1)*ni+l]   : 0.0;
            double below  = (k < ni-1) ? u[(k+1)*ni+l]   : 0.0;
            double left   = (l > 0)    ? u[k*ni+(l-1)]   : 0.0;
            double right  = (l < ni-1) ? u[k*ni+(l+1)]   : 0.0;
            double lap    = above + below + left + right - 4.0*u[k*ni+l];
            double res    = fabs(lap/h2 + src[k*ni+l]);
            if (res > max_res) max_res = res;
        }
    return max_res;
}

int main() {
    const int sz = NI * NI;
    const int full_sz = NX * NX;

    char path[256];
    snprintf(path, sizeof(path), "bench/serial_cache_NX%d.txt", NX);
    FILE *cf = fopen(path, "r");
    if (!cf) { fprintf(stderr, "missing cache file %s -- run serial_cache_gen first\n", path); return 1; }
    int    serial_iter; double serial_res, time_serial, serial_exact_err, u_center_serial;
    if (fscanf(cf, "%d %lf %lf %lf %lf", &serial_iter, &serial_res, &time_serial, &serial_exact_err, &u_center_serial) != 5) {
        fprintf(stderr, "bad cache file %s\n", path); return 1;
    }
    fclose(cf);

    double *h_source  = (double*)calloc(sz, sizeof(double));
    double *h_gpu_sol = (double*)calloc(sz, sizeof(double));
    double *h_u_full  = (double*)calloc(full_sz, sizeof(double));
    double *h_f_full  = (double*)calloc(full_sz, sizeof(double));
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            h_source[k*NI+l] = source_fn(x, y);
            h_f_full[(k+1)*NX + (l+1)] = h_source[k*NI+l];
        }

    double *d_u_old, *d_u_new, *d_f;
    CUDA_CHECK(cudaMalloc(&d_u_old, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_new, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,     full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u_old, 0, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u_new, 0, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_f, h_f_full, full_sz*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((NX + BLOCK_X - 1) / BLOCK_X, (NX + BLOCK_Y - 1) / BLOCK_Y);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int    gpu_iter    = 0;
    double gpu_res     = 1.0;
    float  time_gpu_ms = 0.0f;

    do {
        gpu_iter++;
        CUDA_CHECK(cudaEventRecord(ev_start));
        iteratePoisson<<<grid, block>>>(d_u_old, d_f, d_u_new);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        time_gpu_ms += ms;
        double *tmp = d_u_old; d_u_old = d_u_new; d_u_new = tmp;
        if (gpu_iter % 100 == 0) {
            CUDA_CHECK(cudaMemcpy(h_u_full, d_u_old, full_sz*sizeof(double), cudaMemcpyDeviceToHost));
            for (int k = 0; k < NI; k++)
                for (int l = 0; l < NI; l++)
                    h_gpu_sol[k*NI+l] = h_u_full[(k+1)*NX+(l+1)];
            gpu_res = compute_residual(h_gpu_sol, h_source, NI, H2);
        }
    } while (gpu_res > EPS && gpu_iter < N_ITER_MAX);

    CUDA_CHECK(cudaMemcpy(h_u_full, d_u_old, full_sz*sizeof(double), cudaMemcpyDeviceToHost));
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++)
            h_gpu_sol[k*NI+l] = h_u_full[(k+1)*NX+(l+1)];

    double gpu_exact_err = 0.0;
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            double exact = x*(1.0-x)*y*(1.0-y);
            gpu_exact_err = fmax(gpu_exact_err, fabs(h_gpu_sol[k*NI+l]-exact));
        }

    int l_center = (int)lround(0.5*(NX-1) - 1.0);
    int k_center = l_center;
    if (l_center < 0) l_center = 0; if (l_center > NI-1) l_center = NI-1;
    if (k_center < 0) k_center = 0; if (k_center > NI-1) k_center = NI-1;
    double u_center_gpu = h_gpu_sol[k_center*NI + l_center];

    double gpu_s = time_gpu_ms/1000.0;
    double speedup = time_serial / gpu_s;

    printf("CSVROW,mda,%d,%dx%d,%d,%.6f,%.6f,%.2f,%.6f,%.6f,%.3e\n",
           NX, BLOCK_Y, BLOCK_X, gpu_iter, gpu_s, time_serial,
           speedup, u_center_gpu, u_center_serial, gpu_exact_err);

    if (gpu_iter != serial_iter)
        fprintf(stderr, "NOTE: gpu_iter(%d) != cached serial_iter(%d) at N=%d block=%dx%d\n",
                gpu_iter, serial_iter, NX, BLOCK_Y, BLOCK_X);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaFree(d_u_old); cudaFree(d_u_new); cudaFree(d_f);
    free(h_source); free(h_gpu_sol); free(h_u_full); free(h_f_full);
    return 0;
}
