// Host-to-device bandwidth for the expert slice shapes of -sm tensor offload.
// Source rows of PITCH bytes (iq4_nl down_exps, 640 columns = 360 B), each GPU copies WIDTH bytes of every row.
// Compares cudaMemcpyAsync (1D), cudaMemcpy2DAsync and a gather kernel that reads mapped pinned memory.
// build: nvcc -O2 -o h2d2d h2d2d.cu      run: ./h2d2d
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

static const size_t PITCH = 360;
static const size_t ROWS  = 1310720; // 512 experts x 2560 rows, about 450 MiB

__global__ void gather_rows(const uint32_t * __restrict__ src, uint32_t * __restrict__ dst, size_t rows, size_t wpr, size_t ppr) {
    const size_t n = rows * wpr;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const size_t r = i / wpr;
        const size_t w = i - r * wpr;
        dst[i] = src[r * ppr + w];
    }
}

struct dev_ctx { cudaStream_t s; cudaEvent_t a, b; void * dst; };

static int setup(int d, dev_ctx & c) {
    CHECK(cudaSetDevice(d));
    CHECK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&c.a));
    CHECK(cudaEventCreate(&c.b));
    CHECK(cudaMalloc(&c.dst, PITCH * ROWS));
    return 0;
}

// method 0 = 1D of the full tensor, 1 = 2D, 2 = gather kernel
static void issue(int d, dev_ctx & c, int method, const char * src, size_t width) {
    cudaSetDevice(d);
    cudaEventRecord(c.a, c.s);
    if (method == 0) {
        cudaMemcpyAsync(c.dst, src, PITCH * ROWS, cudaMemcpyHostToDevice, c.s);
    } else if (method == 1) {
        cudaMemcpy2DAsync(c.dst, width, src, PITCH, width, ROWS, cudaMemcpyHostToDevice, c.s);
    } else {
        gather_rows<<<1024, 256, 0, c.s>>>((const uint32_t *) src, (uint32_t *) c.dst, ROWS, width / 4, PITCH / 4);
    }
    cudaEventRecord(c.b, c.s);
}

static double gbps(int d, dev_ctx & c, size_t bytes) {
    cudaSetDevice(d);
    cudaEventSynchronize(c.b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, c.a, c.b);
    return bytes / (ms * 1e6);
}

int main() {
    int n_dev = 0;
    CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) { printf("need 2 GPUs\n"); return 1; }

    char * src = nullptr;
    CHECK(cudaMallocHost((void **) &src, PITCH * ROWS)); // same allocation as the CUDA_Host buffer type
    for (size_t i = 0; i < PITCH * ROWS; i++) { src[i] = (char) i; }

    dev_ctx c[2];
    for (int d = 0; d < 2; d++) { if (setup(d, c[d])) return 1; }

    const char * names[] = { "1D full", "2D", "gather" };
    printf("one GPU at a time (GB/s of useful bytes):\n");
    printf("%-8s %6s %10s %10s\n", "method", "width", "GPU0", "GPU1");
    for (int method = 0; method < 3; method++) {
        const size_t widths_1d[] = { PITCH };
        const size_t widths[]    = { 72, 144, 216, 288, 360 };
        const size_t * w = method == 0 ? widths_1d : widths;
        const int nw = method == 0 ? 1 : 5;
        for (int i = 0; i < nw; i++) {
            double r[2];
            for (int d = 0; d < 2; d++) {
                issue(d, c[d], method, src, w[i]); // warm
                gbps(d, c[d], 1);
                issue(d, c[d], method, src, w[i]);
                r[d] = gbps(d, c[d], w[i] * ROWS);
            }
            printf("%-8s %6zu %10.1f %10.1f\n", names[method], w[i], r[0], r[1]);
        }
    }

    printf("\nboth GPUs at once, 82/18 split of a 360 B row (288 B + 72 B):\n");
    for (int method = 1; method < 3; method++) {
        cudaEvent_t t0, t1;
        cudaSetDevice(0);
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int rep = 0; rep < 2; rep++) {
            cudaSetDevice(0); cudaDeviceSynchronize(); cudaSetDevice(1); cudaDeviceSynchronize();
            cudaSetDevice(0); cudaEventRecord(t0, c[0].s);
            issue(0, c[0], method, src, 288);
            issue(1, c[1], method, src, 72);
            cudaSetDevice(1); cudaStreamSynchronize(c[1].s);
            cudaSetDevice(0); cudaEventRecord(t1, c[0].s); cudaEventSynchronize(t1);
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("%-8s all 360 B rows in %7.1f ms = %5.1f GB/s combined (GPU0 %5.1f, GPU1 %5.1f alone)\n",
            names[method], ms, PITCH * ROWS / (ms * 1e6), gbps(0, c[0], 288 * ROWS), gbps(1, c[1], 72 * ROWS));
    }
    return 0;
}
