// Convergence test: MDA's naive GPU Jacobi kernel vs Serial Jacobi.
//
// Mirrors bench/test_poisson_converge.cu exactly (same NX_GRID parameter,
// same serial baseline code, same residual/convergence logic, same timing
// method) so its numbers are directly comparable to the MDH side. The only
// thing that differs is which kernel does the GPU work: here it's
// mda_kernel.cu's `iteratePoisson` (the Acceleo-generated naive kernel,
// one thread per cell, no tiling), instead of MDH's tuned poisson_1.
//
// Build:
//   nvcc test_poisson_mda.cu mda_kernel.cu -o test_poisson_mda_<N> -DNX_GRID=<N> -DN=<N>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#ifndef NX_GRID
#define NX_GRID 64
#endif
static const int    NX          = NX_GRID;   // full grid side, incl. boundary
static const int    NI          = NX - 2;    // interior side
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

// residual on interior array (NIxNI, oob=0) -- identical to the MDH harness
static double compute_residual(const double *u, const double *src, int ni, double h2) {
    double max_res = 0.0;
    for (int k = 0; k < ni; k++) {
        for (int l = 0; l < ni; l++) {
            double above  = (k > 0)    ? u[(k-1)*ni+l]   : 0.0;
            double below  = (k < ni-1) ? u[(k+1)*ni+l]   : 0.0;
            double left   = (l > 0)    ? u[k*ni+(l-1)]   : 0.0;
            double right  = (l < ni-1) ? u[k*ni+(l+1)]   : 0.0;
            double lap    = above + below + left + right - 4.0*u[k*ni+l];
            double res    = fabs(lap/h2 + src[k*ni+l]);
            if (res > max_res) max_res = res;
        }
    }
    return max_res;
}

int main()
{
    const int sz = NI * NI;       // interior cells, for residual/serial bookkeeping
    const int full_sz = NX * NX;  // full grid, what the MDA kernel operates on

    printf("=== MDA Naive-GPU Convergence Test ===\n");
    printf("Grid: %dx%d  Interior: %dx%d  EPS: %.0e  MaxIter: %d\n\n",
           NX, NX, NI, NI, EPS, N_ITER_MAX);

    double *h_source  = (double*)calloc(sz, sizeof(double));
    double *h_u_old   = (double*)calloc(sz, sizeof(double));
    double *h_u_new   = (double*)calloc(sz, sizeof(double));
    double *h_gpu_sol = (double*)calloc(sz, sizeof(double));

    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            h_source[k*NI+l] = source_fn(x, y);
        }

    // ══════════════════════════════════════════════════════════════════════════
    // PART 1 - SERIAL SOLVER (identical to test_poisson_converge.cu)
    // ══════════════════════════════════════════════════════════════════════════
    printf("--- Serial Jacobi ---\n");

    for (int i = 0; i < sz; i++) h_u_old[i] = h_u_new[i] = 0.0;

    int    serial_iter = 0;
    double serial_res  = 1.0;

    clock_t t_serial_start = clock();

    do {
        serial_iter++;
        for (int k = 0; k < NI; k++)
            for (int l = 0; l < NI; l++) {
                double above = (k > 0)    ? h_u_old[(k-1)*NI+l]   : 0.0;
                double below = (k < NI-1) ? h_u_old[(k+1)*NI+l]   : 0.0;
                double left  = (l > 0)    ? h_u_old[k*NI+(l-1)]   : 0.0;
                double right = (l < NI-1) ? h_u_old[k*NI+(l+1)]   : 0.0;
                h_u_new[k*NI+l] = 0.25*(above+below+left+right+h_source[k*NI+l]*H2);
            }

        if (serial_iter % 100 == 0)
            serial_res = compute_residual(h_u_new, h_source, NI, H2);

        double *tmp = h_u_old; h_u_old = h_u_new; h_u_new = tmp;

        if (serial_iter % 50000 == 0)
            printf("  [serial] iter %7d  res = %e\n", serial_iter, serial_res);

    } while (serial_res > EPS && serial_iter < N_ITER_MAX);

    clock_t t_serial_end = clock();
    double time_serial = (double)(t_serial_end - t_serial_start) / CLOCKS_PER_SEC;

    double serial_exact_err = 0.0;
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            double exact = x*(1.0-x)*y*(1.0-y);
            serial_exact_err = fmax(serial_exact_err, fabs(h_u_old[k*NI+l]-exact));
        }

    printf("\nSerial converged!\n");
    printf("  Iterations        : %d\n", serial_iter);
    printf("  Final residual    : %e\n", serial_res);
    printf("  Time              : %.3f seconds\n", time_serial);
    printf("  Max error vs exact: %e\n\n", serial_exact_err);

    // ══════════════════════════════════════════════════════════════════════════
    // PART 2 - GPU SOLVER (MDA naive kernel, full NxN buffers, boundary=0)
    // ══════════════════════════════════════════════════════════════════════════
    printf("--- GPU Jacobi (MDA-generated naive CUDA) ---\n");

    double *h_u_full = (double*)calloc(full_sz, sizeof(double));
    double *h_f_full = (double*)calloc(full_sz, sizeof(double));
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++)
            h_f_full[(k+1)*NX + (l+1)] = h_source[k*NI+l];

    double *d_u_old, *d_u_new, *d_f;
    CUDA_CHECK(cudaMalloc(&d_u_old, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_new, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,     full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u_old, 0, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u_new, 0, full_sz*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_f, h_f_full, full_sz*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((NX + 31) / 32, (NX + 31) / 32);   // same 32x32 block the ATL rule fixes

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int    gpu_iter    = 0;
    double gpu_res     = 1.0;
    float  time_gpu_ms = 0.0f;

    clock_t t_gpu_wall_start = clock();

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

        if (gpu_iter % 50000 == 0)
            printf("  [gpu]    iter %7d  res = %e\n", gpu_iter, gpu_res);

    } while (gpu_res > EPS && gpu_iter < N_ITER_MAX);

    clock_t t_gpu_wall_end = clock();
    double time_gpu_wall = (double)(t_gpu_wall_end - t_gpu_wall_start) / CLOCKS_PER_SEC;

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

    double max_diff = 0.0;
    for (int i = 0; i < sz; i++)
        max_diff = fmax(max_diff, fabs(h_gpu_sol[i] - h_u_old[i]));
    // NOTE: h_u_old here is the serial solution pointer post-swap from Part 1.

    printf("\nGPU converged!\n");
    printf("  Iterations            : %d\n", gpu_iter);
    printf("  Final residual        : %e\n", gpu_res);
    printf("  GPU kernel time only  : %.3f seconds  (%.3f ms)\n", time_gpu_ms/1000.0f, time_gpu_ms);
    printf("  GPU total wall time   : %.3f seconds  (includes memcpy + residual)\n", time_gpu_wall);
    printf("  Max error vs exact    : %e\n\n", gpu_exact_err);

    printf("=== Comparison ===\n");
    printf("  %-30s  %12s  %12s\n", "Metric", "Serial CPU", "MDA naive GPU");
    printf("  %-30s  %12d  %12d\n", "Iterations to converge", serial_iter, gpu_iter);
    printf("  %-30s  %12e  %12e\n", "Final residual", serial_res, gpu_res);
    printf("  %-30s  %12.3f  %12.3f\n", "Compute time (seconds)", time_serial, time_gpu_ms/1000.0f);
    printf("  %-30s  %12e  %12e\n", "Max error vs exact", serial_exact_err, gpu_exact_err);
    printf("  %-30s  %12s  %12e\n", "Max diff GPU vs serial", "---", max_diff);

    double speedup = time_serial / (time_gpu_ms/1000.0f);
    printf("\n  Speedup (serial / GPU kernel): %.2fx\n\n", speedup);

    if (max_diff < 1e-6)
        printf("MDA naive Poisson is SUCCESSFUL! Solutions match within 1e-6.\n");
    else
        printf("WARNING: GPU and serial solutions differ by %.3e\n", max_diff);

    return 0;
}
