 // ─────────────────────────────────────────────────────────────────────────────
//  PotatoEngine — main.cpp
//
//  Pipeline (all in VRAM, zero CPU↔GPU transfers):
//
//   [Game at 540p]
//       │
//   DXGI Desktop Duplication → D3D11 Texture (VRAM, BGRA8)
//       │
//   cudaGraphicsD3D11RegisterResource  ← zero-copy map
//       │
//   CUDA Lanczos-2 upscale kernel  (540p → 1080p, on GPU)
//   CUDA Sharpening kernel         (adaptive unsharp mask, on GPU)
//   CUDA Saturation kernel         (subtle boost, on GPU)
//       │
//   cudaGraphicsD3D11RegisterResource  ← zero-copy unmap
//       │
//   D3D11 SRV → fullscreen quad → SwapChain::Present
//       │
//   [1080p on your monitor]
//
//  Hotkeys:
//   Alt+F9  — toggle overlay on/off
//   Alt+F10 — increase sharpness (+0.1)
//   Alt+F11 — decrease sharpness (−0.1)
//   Escape  — quit
// ─────────────────────────────────────────────────────────────────────────────

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <cuda_runtime.h>

#include "CaptureEngine.h"
#include "DisplayEngine.h"
#include "UpscaleEngine.cuh"

#include <cstdio>
#include <chrono>
#include <atomic>

// ─── Config ──────────────────────────────────────────────────────────────────
static const int SRC_W = 960;
static const int SRC_H = 540;
static const int DST_W = 1920;
static const int DST_H = 1080;

// ─── Global state (accessed from hotkey thread) ───────────────────────────
static std::atomic<bool>  g_running   { true  };
static std::atomic<bool>  g_active    { true  };
static std::atomic<float> g_sharpness { 0.45f };

// ─── Hotkey listener thread ───────────────────────────────────────────────

static DWORD WINAPI HotkeyThread(LPVOID) {
    // Register hotkeys
    RegisterHotKey(nullptr, 1, MOD_ALT | MOD_NOREPEAT, VK_F9);   // toggle
    RegisterHotKey(nullptr, 2, MOD_ALT | MOD_NOREPEAT, VK_F10);  // sharp+
    RegisterHotKey(nullptr, 3, MOD_ALT | MOD_NOREPEAT, VK_F11);  // sharp-
    RegisterHotKey(nullptr, 4, MOD_NOREPEAT,           VK_ESCAPE);// quit

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        if (msg.message == WM_HOTKEY) {
            switch (msg.wParam) {
            case 1:
                g_active = !g_active;
                printf("[POTATO] Overlay %s\n", g_active.load() ? "ON" : "OFF");
                break;
            case 2: {
                float s = g_sharpness + 0.1f;
                if (s > 1.0f) s = 1.0f;
                g_sharpness = s;
                printf("[POTATO] Sharpness: %.1f\n", s);
                break;
            }
            case 3: {
                float s = g_sharpness - 0.1f;
                if (s < 0.0f) s = 0.0f;
                g_sharpness = s;
                printf("[POTATO] Sharpness: %.1f\n", s);
                break;
            }
            case 4:
                g_running = false;
                break;
            }
        }
    }
    return 0;
}

// ─── CUDA init ───────────────────────────────────────────────────────────────

static bool InitCUDA() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("[POTATO] No CUDA devices found: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("[POTATO] CUDA device: %s\n", prop.name);
    printf("[POTATO] Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("[POTATO] VRAM: %zu MB\n", prop.totalGlobalMem / (1024*1024));

    if (prop.major < 2) {
        printf("[POTATO] WARNING: Compute < 2.0 is very old. "
               "Performance may be limited.\n");
    }
    if (prop.major == 2) {
        printf("[POTATO] Fermi GPU detected. Running in compatibility mode.\n");
        printf("[POTATO] NOTE: For sm_21 (Fermi), make sure you built with "
               "CUDA 8 (see README).\n");
    }

    cudaSetDevice(0);
    return true;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    // Also show a console for debug output
    AllocConsole();
    FILE* f;
    freopen_s(&f, "CONOUT$", "w", stdout);

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        PotatoEngine — CUDA Upscaler          ║\n");
    printf("║    540p → 1080p  |  Lanczos-2 + Sharpen     ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Alt+F9  Toggle overlay on/off               ║\n");
    printf("║  Alt+F10 Increase sharpness (+0.1)           ║\n");
    printf("║  Alt+F11 Decrease sharpness (-0.1)           ║\n");
    printf("║  Escape  Quit                                 ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    // ── CUDA init ─────────────────────────────────────────────────────────
    if (!InitCUDA()) return 1;

    // ── Display engine init (creates D3D11 device we share everywhere) ─────
    DisplayEngine display;
    ID3D11Device*        d3dDevice = nullptr;
    ID3D11DeviceContext* d3dCtx    = nullptr;

    if (!display.Init(DST_W, DST_H, &d3dDevice, &d3dCtx)) {
        printf("[POTATO] DisplayEngine::Init failed\n");
        return 1;
    }

    // ── Capture engine (uses the SAME D3D11 device — required for interop) ──
    CaptureEngine capture;
    if (!capture.Init(d3dDevice, d3dCtx)) {
        printf("[POTATO] CaptureEngine::Init failed\n");
        return 1;
    }

    // ── Create the upscaled output texture (also shared with CUDA) ───────
    D3D11_TEXTURE2D_DESC outDesc{};
    outDesc.Width          = (UINT)DST_W;
    outDesc.Height         = (UINT)DST_H;
    outDesc.MipLevels      = 1;
    outDesc.ArraySize      = 1;
    outDesc.Format         = DXGI_FORMAT_B8G8R8A8_UNORM;
    outDesc.SampleDesc     = { 1, 0 };
    outDesc.Usage          = D3D11_USAGE_DEFAULT;
    outDesc.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    outDesc.MiscFlags      = D3D11_RESOURCE_MISC_SHARED;

    ID3D11Texture2D* outputTex = nullptr;
    if (FAILED(d3dDevice->CreateTexture2D(&outDesc, nullptr, &outputTex))) {
        printf("[POTATO] Failed to create output texture\n");
        return 1;
    }

    // ── Upscale engine ────────────────────────────────────────────────────
    UpscaleConfig upCfg;
    upCfg.srcW      = capture.GetWidth();   // actual desktop width (may != 960)
    upCfg.srcH      = capture.GetHeight();
    upCfg.dstW      = DST_W;
    upCfg.dstH      = DST_H;
    upCfg.sharpness = g_sharpness;

    UpscaleEngine upscaler(upCfg);
    if (!upscaler.RegisterD3DResources(capture.GetCaptureTex(), outputTex)) {
        printf("[POTATO] Failed to register D3D resources with CUDA\n");
        return 1;
    }

    // ── Hotkey listener ───────────────────────────────────────────────────
    HANDLE hHotkeyThread = CreateThread(nullptr, 0, HotkeyThread,
                                        nullptr, 0, nullptr);

    // ── Frame timing ──────────────────────────────────────────────────────
    using Clock = std::chrono::high_resolution_clock;
    auto   lastFPS  = Clock::now();
    int    frames   = 0;

    printf("[POTATO] Pipeline running. Ctrl+C or Escape to quit.\n\n");

    // ══ Main loop ══════════════════════════════════════════════════════════
    while (g_running) {
        if (!display.PumpMessages()) break;

        if (!g_active) {
            Sleep(16);
            continue;
        }

        // 1. Capture desktop frame into VRAM texture
        //    (non-blocking: if no new frame, we re-upscale the previous one)
        capture.AcquireFrame(0); // 0 = non-blocking

        // 2. Sync sharpness from hotkey thread
        upscaler.GetConfig(); // readonly; we rebuild the upCfg inline:
        {
            // Cast away const for sharpness update
            // (safe: only one writer thread, one reader thread)
            const_cast<UpscaleConfig&>(upscaler.GetConfig()).sharpness
                = g_sharpness.load(std::memory_order_relaxed);
        }

        // 3. Run CUDA pipeline (Lanczos-2 + sharpen + saturate) — VRAM only
        upscaler.Upscale();

        // 4. Release DXGI frame BEFORE Present (avoids DXGI timeout)
        capture.ReleaseFrame();

        // 5. Blit upscaled texture to screen via D3D11
        display.Present(outputTex);

        // 6. FPS counter
        ++frames;
        auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - lastFPS).count();
        if (elapsed >= 2.0) {
            printf("[POTATO] FPS: %.1f  |  Sharpness: %.2f  |  %dx%d → %dx%d\n",
                   frames / elapsed,
                   g_sharpness.load(),
                   upCfg.srcW, upCfg.srcH, DST_W, DST_H);
            frames  = 0;
            lastFPS = now;
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    printf("[POTATO] Shutting down...\n");
    upscaler.ReleaseD3DResources();
    if (outputTex) outputTex->Release();
    capture.Shutdown();
    display.Shutdown();

    if (hHotkeyThread) {
        PostThreadMessage(GetThreadId(hHotkeyThread), WM_QUIT, 0, 0);
        WaitForSingleObject(hHotkeyThread, 1000);
        CloseHandle(hHotkeyThread);
    }

    printf("[POTATO] Goodbye, potato warrior.\n");
    return 0;
}
