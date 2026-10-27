// Serial Jacobi baseline, run once per grid size and cached to disk, so the
// GPU block-size sweep (bench_gpu_sweep_mdh.cu / bench_gpu_sweep_mda.cu) does
// not have to re-run the O(N^4) serial solver once per block configuration.
// Identical math to the serial section already in test_poisson_converge.cu /
// test_poisson_mda.cu -- this file only adds caching and a u(0.5,0.5) readout.
//
// Build:
//   g++ -O2 -DNX_GRID=<N> bench/serial_cache_gen.cc -o /tmp/serial_gen_<N>
// Run:
//   /tmp/serial_gen_<N>
// Writes bench/serial_cache_NX<N>.txt : iter res time exact_err u_center

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#ifndef NX_GRID
#define NX_GRID 64
#endif
static const int    NX          = NX_GRID;
static const int    NI          = NX - 2;
static const double H           = 1.0 / (NX - 1.0);
static const double H2          = H * H;
static const double EPS         = 1e-4;
static const int    N_ITER_MAX  = 3000000;

static inline double source_fn(double x, double y) {
    return 2.0 * (x*(1.0-x) + y*(1.0-y));
}

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

int main() {
    const int sz = NI * NI;
    double *h_source = (double*)calloc(sz, sizeof(double));
    double *h_u_old  = (double*)calloc(sz, sizeof(double));
    double *h_u_new  = (double*)calloc(sz, sizeof(double));

    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            h_source[k*NI+l] = source_fn(x, y);
        }

    int    serial_iter = 0;
    double serial_res  = 1.0;
    clock_t t0 = clock();

    do {
        serial_iter++;
        for (int k = 0; k < NI; k++)
            for (int l = 0; l < NI; l++) {
                double above = (k > 0)    ? h_u_old[(k-1)*NI+l] : 0.0;
                double below = (k < NI-1) ? h_u_old[(k+1)*NI+l] : 0.0;
                double left  = (l > 0)    ? h_u_old[k*NI+(l-1)] : 0.0;
                double right = (l < NI-1) ? h_u_old[k*NI+(l+1)] : 0.0;
                h_u_new[k*NI+l] = 0.25*(above+below+left+right+h_source[k*NI+l]*H2);
            }
        if (serial_iter % 100 == 0)
            serial_res = compute_residual(h_u_new, h_source, NI, H2);
        double *tmp = h_u_old; h_u_old = h_u_new; h_u_new = tmp;
        if (serial_iter % 200000 == 0)
            fprintf(stderr, "  [serial N=%d] iter %7d  res = %e\n", NX, serial_iter, serial_res);
    } while (serial_res > EPS && serial_iter < N_ITER_MAX);

    clock_t t1 = clock();
    double time_serial = (double)(t1 - t0) / CLOCKS_PER_SEC;

    double exact_err = 0.0;
    for (int k = 0; k < NI; k++)
        for (int l = 0; l < NI; l++) {
            double x = (l+1)*H, y = (k+1)*H;
            double exact = x*(1.0-x)*y*(1.0-y);
            exact_err = fmax(exact_err, fabs(h_u_old[k*NI+l]-exact));
        }

    // interior grid point nearest the physical center (0.5, 0.5)
    int l_center = (int)lround(0.5*(NX-1) - 1.0);
    int k_center = l_center;
    if (l_center < 0) l_center = 0; if (l_center > NI-1) l_center = NI-1;
    if (k_center < 0) k_center = 0; if (k_center > NI-1) k_center = NI-1;
    double u_center = h_u_old[k_center*NI + l_center];

    fprintf(stderr, "N=%d  iter=%d  res=%e  time=%.6f s  exact_err=%e  u_center=%.6f\n",
            NX, serial_iter, serial_res, time_serial, exact_err, u_center);

    char path[256];
    snprintf(path, sizeof(path), "bench/serial_cache_NX%d.txt", NX);
    FILE *f = fopen(path, "w");
    fprintf(f, "%d %.10e %.10f %.10e %.10f\n", serial_iter, serial_res, time_serial, exact_err, u_center);
    fclose(f);

    free(h_source); free(h_u_old); free(h_u_new);
    return 0;
}
