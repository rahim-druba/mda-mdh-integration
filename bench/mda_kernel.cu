// Extracted verbatim (math/logic unchanged) from the Acceleo-generated
// MDA poisson/Poisson_gen/poisson2code/output/poisson.cu - see
// mda_reference/poisson_mda_original.cu for the untouched original file.
//
// The ONLY change from the original: N and h were hardcoded `#define`s fixed
// to N=1024. Here they're overridable at compile time (-DN=64/128/256) so the
// exact same kernel body can be benchmarked at the same grid sizes used for
// the MDH side, instead of being locked to one size. Not one line of the
// actual stencil math below was touched.

#ifndef N
#define N 1024
#endif
#define h (1.0 / (N - 1.0))

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
