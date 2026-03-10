// ─────────────────────────────────────────────────────────────────────────────
//  CaptureEngine.cpp  —  DXGI Desktop Duplication
//  Captures the screen into a GPU texture.  No CPU copies.
// ─────────────────────────────────────────────────────────────────────────────

#include "CaptureEngine.h"
#include <cstdio>
#include <stdexcept>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

CaptureEngine::~CaptureEngine() { Shutdown(); }

bool CaptureEngine::Init(ID3D11Device* device, ID3D11DeviceContext* ctx) {
    device_ = device;
    ctx_    = ctx;

    // ── Get the primary DXGI output ──────────────────────────────────────────
    ComPtr<IDXGIDevice>  dxgiDevice;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<IDXGIOutput>  output;
    ComPtr<IDXGIOutput1> output1;

    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)))) {
        printf("[POTATO] QueryInterface IDXGIDevice failed\n");
        return false;
    }
    if (FAILED(dxgiDevice->GetAdapter(&adapter))) {
        printf("[POTATO] GetAdapter failed\n");
        return false;
    }
    if (FAILED(adapter->EnumOutputs(0, &output))) {
        printf("[POTATO] EnumOutputs failed — no monitor?\n");
        return false;
    }
    if (FAILED(output->QueryInterface(IID_PPV_ARGS(&output1)))) {
        printf("[POTATO] QueryInterface IDXGIOutput1 failed\n");
        return false;
    }

    // ── Grab output description for dimensions ───────────────────────────────
    DXGI_OUTPUT_DESC outDesc{};
    output->GetDesc(&outDesc);
    width_  = outDesc.DesktopCoordinates.right  - outDesc.DesktopCoordinates.left;
    height_ = outDesc.DesktopCoordinates.bottom - outDesc.DesktopCoordinates.top;
    printf("[POTATO] Desktop size: %dx%d\n", width_, height_);

    // ── Create output duplication ────────────────────────────────────────────
    HRESULT hr = output1->DuplicateOutput(device, &duplication_);
    if (FAILED(hr)) {
        printf("[POTATO] DuplicateOutput failed: 0x%08X\n"
               "         (run as Administrator, or disable HDR)\n", hr);
        return false;
    }

    // ── Create persistent VRAM texture for the captured frame ────────────────
    D3D11_TEXTURE2D_DESC td{};
    td.Width          = (UINT)width_;
    td.Height         = (UINT)height_;
    td.MipLevels      = 1;
    td.ArraySize      = 1;
    td.Format         = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc     = { 1, 0 };
    td.Usage          = D3D11_USAGE_DEFAULT;
    td.BindFlags      = D3D11_BIND_SHADER_RESOURCE; // CUDA interop needs this
    td.CPUAccessFlags = 0;
    td.MiscFlags      = D3D11_RESOURCE_MISC_SHARED; // allows CUDA access

    hr = device->CreateTexture2D(&td, nullptr, captureTex_.GetAddressOf());
    if (FAILED(hr)) {
        printf("[POTATO] CreateTexture2D (capture) failed: 0x%08X\n", hr);
        return false;
    }

    printf("[POTATO] CaptureEngine ready. DXGI Desktop Duplication active.\n");
    return true;
}

bool CaptureEngine::AcquireFrame(UINT timeoutMs) {
    if (!duplication_) return false;

    // Release previous frame first if still held
    if (hasFrame_) ReleaseFrame();

    ComPtr<IDXGIResource>          dxgiRes;
    DXGI_OUTDUPL_FRAME_INFO        frameInfo{};

    HRESULT hr = duplication_->AcquireNextFrame(timeoutMs, &frameInfo, &dxgiRes);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT)  return false; // no new frame yet
    if (hr == DXGI_ERROR_ACCESS_LOST) {
        // Desktop mode changed (resolution, HDR toggle…) — reinit needed
        printf("[POTATO] Desktop duplication access lost — reinit required\n");
        duplication_.Reset();
        return false;
    }
    if (FAILED(hr)) {
        printf("[POTATO] AcquireNextFrame failed: 0x%08X\n", hr);
        return false;
    }

    hasFrame_ = true;

    // ── Copy the acquired frame into our persistent CUDA-registered texture ──
    // This is a GPU→GPU copy inside VRAM: zero CPU involvement.
    ComPtr<ID3D11Texture2D> frameTex;
    hr = dxgiRes.As(&frameTex);
    if (SUCCEEDED(hr)) {
        ctx_->CopyResource(captureTex_.Get(), frameTex.Get());
    }

    return true;
}

void CaptureEngine::ReleaseFrame() {
    if (hasFrame_ && duplication_) {
        duplication_->ReleaseFrame();
        hasFrame_ = false;
    }
}

void CaptureEngine::Shutdown() {
    ReleaseFrame();
    duplication_.Reset();
    captureTex_.Reset();
}
