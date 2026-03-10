// ─────────────────────────────────────────────────────────────────────────────
//  PotatoEngine  —  UpscaleEngine.cu
//
//  All GPU kernels.  Zero host/device transfers — reads & writes D3D surfaces
//  directly through CUDA surface objects (data stays in VRAM the whole time).
//
//  ── Fermi (sm_21) compatibility ──────────────────────────────────────────
//  • surf2Dread / surf2Dwrite : available since CUDA 5 / sm_20  ✓
//  • cudaSurfaceObject_t      : available since CUDA 5 / sm_20  ✓
//  • No __ldg(), no shuffles (those are sm_30+). Not used here. ✓
//  • Max registers/thread: 63 on Fermi — __launch_bounds__ used everywhere
//    to prevent register spilling to slow local memory.
//  • 16×16 = 256 threads/block: safe for Fermi, good occupancy
// ─────────────────────────────────────────────────────────────────────────────

#include "UpscaleEngine.cuh"
#include <cuda_runtime.h>
#include <surface_functions.h>
#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
// Tells NVCC to cap register allocation so Fermi's 63-register limit
// is respected.  Prevents spilling to local memory (very slow on DDR3).
#define FERMI_LIGHT  __launch_bounds__(256, 2)   // sharpen / saturation
#define FERMI_HEAVY  __launch_bounds__(128, 2)   // Lanczos (most registers)

// ─── Device helpers ──────────────────────────────────────────────────────────

__device__ __forceinline__ float d_sinc(float x) {
    if (fabsf(x) < 1e-7f) return 1.0f;
    float px = M_PI * x;
    return __sinf(px) / px;               // __sinf = fast hardware sin, sm_20+
}

__device__ __forceinline__ float lanczos2(float x) {
    x = fabsf(x);
    if (x >= 2.0f) return 0.0f;
    return d_sinc(x) * d_sinc(x * 0.5f);
}

__device__ __forceinline__ unsigned char clamp_uc(float v) {
    return (unsigned char)__float2int_rn(fminf(fmaxf(v, 0.0f), 255.0f));
}

// ─── Kernel 1: Lanczos-2 Upscale ─────────────────────────────────────────────
//
//  Each output pixel reads a 4×4 neighbourhood from the source surface,
//  weights the samples by the Lanczos-2 kernel, and writes the result.
//  surf2Dread / surf2Dwrite keep everything in VRAM — no RAM transfers.
// ─────────────────────────────────────────────────────────────────────────────

__global__ FERMI_HEAVY void k_lanczos2_upscale(
    cudaSurfaceObject_t src,
    cudaSurfaceObject_t dst,
    int srcW, int srcH,
    int dstW, int dstH)
{
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;

    // Back-project destination pixel to source space (pixel-center correction)
    const float sx = (dx + 0.5f) * (float)srcW / (float)dstW - 0.5f;
    const float sy = (dy + 0.5f) * (float)srcH / (float)dstH - 0.5f;
    const int   ix = (int)floorf(sx);
    const int   iy = (int)floorf(sy);

    float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f, wsum = 0.f;

    // 4×4 Lanczos-2 neighbourhood
    // NOTE: unrolled manually for Fermi — #pragma unroll on a loop
    //       with function calls can bloat registers over the 63 limit.
    int ky, kx;
    for (ky = -1; ky <= 2; ky++) {
        float wy = lanczos2(sy - (float)(iy + ky));
        for (kx = -1; kx <= 2; kx++) {
            float w = wy * lanczos2(sx - (float)(ix + kx));
            if (fabsf(w) < 1e-9f) continue;

            int px = min(max(ix + kx, 0), srcW - 1);
            int py = min(max(iy + ky, 0), srcH - 1);

            uchar4 p;
            surf2Dread(&p, src, px * (int)sizeof(uchar4), py);

            acc_r += (float)p.x * w;
            acc_g += (float)p.y * w;
            acc_b += (float)p.z * w;
            wsum  += w;
        }
    }

    if (wsum > 1e-7f) {
        float inv = 1.f / wsum;
        acc_r *= inv;
        acc_g *= inv;
        acc_b *= inv;
    }

    uchar4 out;
    out.x = clamp_uc(acc_r);
    out.y = clamp_uc(acc_g);
    out.z = clamp_uc(acc_b);
    out.w = 255;

    surf2Dwrite(out, dst, dx * (int)sizeof(uchar4), dy);
}

// ─── Kernel 2: Adaptive Unsharp-Mask Sharpening ──────────────────────────────
//
//  Laplacian sharpening: out = (1+k)*center − (k/4)*(N+S+E+W)
//  In-place on the upscaled surface.
// ─────────────────────────────────────────────────────────────────────────────

__global__ FERMI_LIGHT void k_sharpen(
    cudaSurfaceObject_t surf,
    int width, int height,
    float amount)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    // Skip border pixels (no valid neighbours)
    if (cx <= 0 || cy <= 0 || cx >= width - 1 || cy >= height - 1) return;

    uchar4 cc, cn, cs, ce, cw;
    surf2Dread(&cc, surf,  cx      * (int)sizeof(uchar4), cy);
    surf2Dread(&cn, surf,  cx      * (int)sizeof(uchar4), cy - 1);
    surf2Dread(&cs, surf,  cx      * (int)sizeof(uchar4), cy + 1);
    surf2Dread(&ce, surf, (cx + 1) * (int)sizeof(uchar4), cy);
    surf2Dread(&cw, surf, (cx - 1) * (int)sizeof(uchar4), cy);

    float k  = amount * 0.25f;
    float kc = 1.f + amount;

    uchar4 out;
    out.x = clamp_uc(kc*(float)cc.x - k*((float)cn.x+(float)cs.x+(float)ce.x+(float)cw.x));
    out.y = clamp_uc(kc*(float)cc.y - k*((float)cn.y+(float)cs.y+(float)ce.y+(float)cw.y));
    out.z = clamp_uc(kc*(float)cc.z - k*((float)cn.z+(float)cs.z+(float)ce.z+(float)cw.z));
    out.w = 255;

    surf2Dwrite(out, surf, cx * (int)sizeof(uchar4), cy);
}

// ─── Kernel 3: Saturation boost (luma-weighted) ──────────────────────────────

__global__ FERMI_LIGHT void k_saturation(
    cudaSurfaceObject_t surf,
    int width, int height,
    float sat)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 p;
    surf2Dread(&p, surf, x * (int)sizeof(uchar4), y);

    float r = (float)p.x * (1.f/255.f);
    float g = (float)p.y * (1.f/255.f);
    float b = (float)p.z * (1.f/255.f);

    // Rec. 709 luma weights
    float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;

    r = luma + sat * (r - luma);
    g = luma + sat * (g - luma);
    b = luma + sat * (b - luma);

    uchar4 out;
    out.x = clamp_uc(r * 255.f);
    out.y = clamp_uc(g * 255.f);
    out.z = clamp_uc(b * 255.f);
    out.w = 255;

    surf2Dwrite(out, surf, x * (int)sizeof(uchar4), y);
}

// ─── Launcher wrappers ───────────────────────────────────────────────────────
//  Block size 16×16 = 256 threads: good occupancy on Fermi.
//  The heavy Lanczos kernel uses 8×16 = 128 threads to stay under
//  the 63 register limit (more registers per thread, fewer threads per block).

void LaunchLanczos2Kernel(cudaSurfaceObject_t src,
                          cudaSurfaceObject_t dst,
                          int srcW, int srcH,
                          int dstW, int dstH,
                          cudaStream_t stream)
{
    // 8×16 = 128 threads/block for Fermi register budget
    dim3 block(8, 16);
    dim3 grid((dstW + 7) / 8, (dstH + 15) / 16);
    k_lanczos2_upscale<<<grid, block, 0, stream>>>(
        src, dst, srcW, srcH, dstW, dstH);
}

void LaunchSharpenKernel(cudaSurfaceObject_t surf,
                         int width, int height,
                         float amount,
                         cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_sharpen<<<grid, block, 0, stream>>>(surf, width, height, amount);
}

void LaunchSaturationKernel(cudaSurfaceObject_t surf,
                            int width, int height,
                            float saturation,
                            cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_saturation<<<grid, block, 0, stream>>>(surf, width, height, saturation);
}

// ─── UpscaleEngine implementation ────────────────────────────────────────────

UpscaleEngine::UpscaleEngine(const UpscaleConfig& cfg) : cfg_(cfg) {}

UpscaleEngine::~UpscaleEngine() {
    ReleaseD3DResources();
}

bool UpscaleEngine::RegisterD3DResources(ID3D11Texture2D* srcTex,
                                          ID3D11Texture2D* dstTex)
{
    cudaError_t err;

    err = cudaGraphicsD3D11RegisterResource(
        &srcRes_, srcTex,
        cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        printf("[POTATO] Register src failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaGraphicsD3D11RegisterResource(
        &dstRes_, dstTex,
        cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        printf("[POTATO] Register dst failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    printf("[POTATO] D3D11<->CUDA zero-copy interop registered.\n");
    return true;
}

bool UpscaleEngine::MapSurfaces() {
    cudaGraphicsResource_t res[2] = { srcRes_, dstRes_ };
    if (cudaGraphicsMapResources(2, res, 0) != cudaSuccess) return false;

    auto makeSurf = [](cudaGraphicsResource_t r) -> cudaSurfaceObject_t {
        cudaArray_t arr;
        cudaGraphicsSubResourceGetMappedArray(&arr, r, 0, 0);
        cudaResourceDesc rd{};
        rd.resType         = cudaResourceTypeArray;
        rd.res.array.array = arr;
        cudaSurfaceObject_t s = 0;
        cudaCreateSurfaceObject(&s, &rd);
        return s;
    };

    srcSurf_ = makeSurf(srcRes_);
    dstSurf_ = makeSurf(dstRes_);
    return (srcSurf_ != 0 && dstSurf_ != 0);
}

void UpscaleEngine::UnmapSurfaces() {
    if (srcSurf_) { cudaDestroySurfaceObject(srcSurf_); srcSurf_ = 0; }
    if (dstSurf_) { cudaDestroySurfaceObject(dstSurf_); dstSurf_ = 0; }
    cudaGraphicsResource_t res[2] = { srcRes_, dstRes_ };
    cudaGraphicsUnmapResources(2, res, 0);
}

bool UpscaleEngine::Upscale() {
    if (!srcRes_ || !dstRes_) return false;
    if (!MapSurfaces()) return false;

    // All three kernels on the default stream, sequential
    LaunchLanczos2Kernel(srcSurf_, dstSurf_,
                         cfg_.srcW, cfg_.srcH,
                         cfg_.dstW, cfg_.dstH, 0);

    if (cfg_.sharpness > 0.01f)
        LaunchSharpenKernel(dstSurf_,
                            cfg_.dstW, cfg_.dstH,
                            cfg_.sharpness, 0);

    if (cfg_.saturation != 1.0f)
        LaunchSaturationKernel(dstSurf_,
                               cfg_.dstW, cfg_.dstH,
                               cfg_.saturation, 0);

    // Fermi note: cudaStreamSynchronize(0) is cheap here —
    // we're CPU-bound on the DXGI Present anyway.
    cudaStreamSynchronize(0);

    UnmapSurfaces();
    return (cudaGetLastError() == cudaSuccess);
}

void UpscaleEngine::ReleaseD3DResources() {
    if (srcRes_) { cudaGraphicsUnregisterResource(srcRes_); srcRes_ = nullptr; }
    if (dstRes_) { cudaGraphicsUnregisterResource(dstRes_); dstRes_ = nullptr; }
}
