// GPU-only block-size sweep harness for the MDH-side kernels (Integrated via
// bridge, or MDH-alone hand-written spec -- both expose the same `poisson_1`
// symbol, so this one file links against either generated/poisson_1.cu or
// generated/handwritten/poisson_1.cu depending on the build command).
//
// Reads the serial baseline from bench/serial_cache_NX<N>.txt (produced once
// per grid size by serial_cache_gen.cc) instead of re-running the O(N^4)
// serial solver for every block configuration swept.
//
// Build (example, 16x16 block at N=256):
//   nvcc bench/bench_gpu_sweep_mdh.cu generated/poisson_1.cu \
//     -o /tmp/sweep_integrated_256_16x16 \
//     -DNX_GRID=256 -DTYPE_T=double -DTYPE_TS=double \
//     -DCACHE_L_CB=0 -DCACHE_P_CB=0 -DG_CB_RES_DEST_LEVEL=2 \
//     -DG_CB_SIZE_L_1=254 -DG_CB_SIZE_L_2=254 \
//     -DL_CB_RES_DEST_LEVEL=1 -DL_CB_SIZE_L_1=16 -DL_CB_SIZE_L_2=16 \
//     -DNUM_WG_L_1=16 -DNUM_WG_L_2=16 -DNUM_WI_L_1=16 -DNUM_WI_L_2=16 \
//     -DOCL_DIM_L_1=1 -DOCL_DIM_L_2=0 \
//     -DP_CB_RES_DEST_LEVEL=0 -DP_CB_SIZE_L_1=1 -DP_CB_SIZE_L_2=1
//
// Prints one CSV line: label,N,block,iterations,gpu_s,cpu_s,speedup,u_center_gpu,u_center_serial,gpu_exact_err

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#ifndef NX_GRID
#define NX_GRID 64
#endif
#ifndef SWEEP_LABEL
#define SWEEP_LABEL "mdh"
#endif

static const int    NX          = NX_GRID;
static const int    NI          = NX - 2;
static const double H           = 1.0 / (NX - 1.0);
static const double H2          = H * H;
static const double EPS         = 1e-4;
static const int    N_ITER_MAX  = 3000000;

static const dim3 BLOCK1 { NUM_WI_L_2, NUM_WI_L_1, 1 };
static const dim3 GRID1  { NUM_WG_L_2, NUM_WG_L_1, 1 };

extern __global__ void poisson_1(
    double const * const __restrict__ U,
    double const * const __restrict__ SOURCE,
    const double                      h2,
    double       * const __restrict__ res_g,
    double       * const __restrict__ int_res
);

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

    // read cached serial baseline for this NX
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
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            h_source[k*NI+l] = source_fn(x, y);
        }

    double *d_U, *d_int_res, *d_res_g, *d_SOURCE;
    CUDA_CHECK(cudaMalloc(&d_U,       sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_int_res, sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_res_g,   sz*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SOURCE,  sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_U,       0, sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_int_res, 0, sz*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_res_g,   0, sz*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_SOURCE, h_source, sz*sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    int    gpu_iter    = 0;
    double gpu_res     = 1.0;
    float  time_gpu_ms = 0.0f;

    do {
        gpu_iter++;
        CUDA_CHECK(cudaEventRecord(ev_start));
        poisson_1<<<GRID1, BLOCK1>>>(d_U, d_SOURCE, H2, d_res_g, d_int_res);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        time_gpu_ms += ms;
        double *tmp = d_U; d_U = d_int_res; d_int_res = tmp;
        if (gpu_iter % 100 == 0) {
            CUDA_CHECK(cudaMemcpy(h_gpu_sol, d_U, sz*sizeof(double), cudaMemcpyDeviceToHost));
            gpu_res = compute_residual(h_gpu_sol, h_source, NI, H2);
        }
    } while (gpu_res > EPS && gpu_iter < N_ITER_MAX);

    CUDA_CHECK(cudaMemcpy(h_gpu_sol, d_U, sz*sizeof(double), cudaMemcpyDeviceToHost));

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

    printf("CSVROW,%s,%d,%dx%d,%d,%.6f,%.6f,%.2f,%.6f,%.6f,%.3e\n",
           SWEEP_LABEL, NX, NUM_WI_L_1, NUM_WI_L_2, gpu_iter, gpu_s, time_serial,
           speedup, u_center_gpu, u_center_serial, gpu_exact_err);

    if (gpu_iter != serial_iter)
        fprintf(stderr, "NOTE: gpu_iter(%d) != cached serial_iter(%d) at N=%d block=%dx%d\n",
                gpu_iter, serial_iter, NX, NUM_WI_L_1, NUM_WI_L_2);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaFree(d_U); cudaFree(d_int_res); cudaFree(d_res_g); cudaFree(d_SOURCE);
    free(h_source); free(h_gpu_sol);
    return 0;
}
