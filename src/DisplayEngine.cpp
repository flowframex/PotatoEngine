// ─────────────────────────────────────────────────────────────────────────────
//  DisplayEngine.cpp  —  D3D11 borderless-fullscreen display
//  Blits the upscaled VRAM texture to the swap-chain with a single draw call.
// ─────────────────────────────────────────────────────────────────────────────

#include "DisplayEngine.h"
#include <d3dcompiler.h>
#include <cstdio>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// ─── Inline HLSL shaders ─────────────────────────────────────────────────────

static const char* s_hlsl = R"HLSL(
// Vertex: full-screen quad (positions baked in, no VB needed via SV_VertexID)
struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD;
};

VSOut VS(uint id : SV_VertexID)
{
    // Two triangles making a fullscreen quad:
    // id 0:(−1,−1,0,1) uv(0,1)
    // id 1:(−1, 1,0,1) uv(0,0)
    // id 2:( 1,−1,0,1) uv(1,1)
    // id 3:( 1, 1,0,1) uv(1,0)
    float2 pos = float2((id & 2) ? 1.0 : -1.0,
                        (id & 1) ? 1.0 : -1.0);
    VSOut o;
    o.pos = float4(pos, 0, 1);
    o.uv  = float2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return o;
}

Texture2D    gTex     : register(t0);
SamplerState gSampler : register(s0);

float4 PS(VSOut i) : SV_TARGET
{
    return gTex.Sample(gSampler, i.uv);
}
)HLSL";

// ─── WndProc ─────────────────────────────────────────────────────────────────

LRESULT CALLBACK DisplayEngine::WndProc(HWND hwnd, UINT msg,
                                         WPARAM wp, LPARAM lp)
{
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    if (msg == WM_KEYDOWN && wp == VK_ESCAPE) { DestroyWindow(hwnd); return 0; }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// ─── Init ────────────────────────────────────────────────────────────────────

bool DisplayEngine::Init(int width, int height,
                          ID3D11Device**        outDevice,
                          ID3D11DeviceContext** outCtx)
{
    width_  = width;
    height_ = height;

    // ── Register window class ────────────────────────────────────────────────
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"PotatoEngineWindow";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    // ── Borderless fullscreen window ─────────────────────────────────────────
    hwnd_ = CreateWindowExW(
        WS_EX_TOPMOST,
        L"PotatoEngineWindow", L"PotatoEngine — GT730 Upscaler",
        WS_POPUP,
        0, 0, width_, height_,
        nullptr, nullptr,
        GetModuleHandleW(nullptr), nullptr);

    if (!hwnd_) {
        printf("[POTATO] CreateWindowExW failed\n");
        return false;
    }
    ShowWindow(hwnd_, SW_SHOW);

    // ── D3D11 + swap chain ───────────────────────────────────────────────────
    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount                        = 2;
    scd.BufferDesc.Width                   = (UINT)width_;
    scd.BufferDesc.Height                  = (UINT)height_;
    scd.BufferDesc.Format                  = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.BufferDesc.RefreshRate.Numerator   = 0; // use vsync/monitor rate
    scd.BufferDesc.RefreshRate.Denominator = 1;
    scd.BufferUsage                        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow                       = hwnd_;
    scd.SampleDesc                         = { 1, 0 };
    scd.Windowed                           = TRUE;
    scd.SwapEffect                         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Flags                              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    D3D_FEATURE_LEVEL gotLevel{};

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        featureLevels, ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &scd,
        swapChain_.GetAddressOf(),
        device_.GetAddressOf(),
        &gotLevel,
        ctx_.GetAddressOf());

    if (FAILED(hr)) {
        printf("[POTATO] D3D11CreateDeviceAndSwapChain failed: 0x%08X\n", hr);
        return false;
    }
    printf("[POTATO] D3D11 device created. Feature level: 0x%04X\n", gotLevel);

    // ── Render target view for back buffer ───────────────────────────────────
    ComPtr<ID3D11Texture2D> backBuf;
    swapChain_->GetBuffer(0, IID_PPV_ARGS(&backBuf));
    device_->CreateRenderTargetView(backBuf.Get(), nullptr,
                                    rtv_.GetAddressOf());

    // ── Viewport ─────────────────────────────────────────────────────────────
    D3D11_VIEWPORT vp{};
    vp.Width    = (float)width_;
    vp.Height   = (float)height_;
    vp.MaxDepth = 1.0f;
    ctx_->RSSetViewports(1, &vp);

    if (!CreateShadersAndQuad()) return false;

    // ── Point sampler (CUDA already applied quality filtering) ───────────────
    D3D11_SAMPLER_DESC sd{};
    sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    device_->CreateSamplerState(&sd, sampler_.GetAddressOf());
    ctx_->PSSetSamplers(0, 1, sampler_.GetAddressOf());

    *outDevice = device_.Get();
    *outCtx    = ctx_.Get();

    printf("[POTATO] DisplayEngine ready. Window: %dx%d\n", width_, height_);
    return true;
}

bool DisplayEngine::CreateShadersAndQuad() {
    ComPtr<ID3DBlob> vsBlob, psBlob, errBlob;

    // Compile VS
    HRESULT hr = D3DCompile(s_hlsl, strlen(s_hlsl), nullptr, nullptr, nullptr,
                            "VS", "vs_4_0", 0, 0,
                            vsBlob.GetAddressOf(), errBlob.GetAddressOf());
    if (FAILED(hr)) {
        if (errBlob)
            printf("[POTATO] VS compile error: %s\n",
                   (char*)errBlob->GetBufferPointer());
        return false;
    }

    // Compile PS
    hr = D3DCompile(s_hlsl, strlen(s_hlsl), nullptr, nullptr, nullptr,
                    "PS", "ps_4_0", 0, 0,
                    psBlob.GetAddressOf(), errBlob.GetAddressOf());
    if (FAILED(hr)) {
        if (errBlob)
            printf("[POTATO] PS compile error: %s\n",
                   (char*)errBlob->GetBufferPointer());
        return false;
    }

    device_->CreateVertexShader(vsBlob->GetBufferPointer(),
                                vsBlob->GetBufferSize(),
                                nullptr, vs_.GetAddressOf());
    device_->CreatePixelShader(psBlob->GetBufferPointer(),
                               psBlob->GetBufferSize(),
                               nullptr, ps_.GetAddressOf());

    ctx_->VSSetShader(vs_.Get(), nullptr, 0);
    ctx_->PSSetShader(ps_.Get(), nullptr, 0);
    ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    return true;
}

// ─── Present ─────────────────────────────────────────────────────────────────

void DisplayEngine::Present(ID3D11Texture2D* upscaledTex) {
    // Create or re-create SRV for this texture
    // (In a tight loop the texture pointer won't change, but guard anyway)
    static ID3D11Texture2D* lastTex = nullptr;
    if (upscaledTex != lastTex) {
        srv_.Reset();
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
        srvd.Format              = DXGI_FORMAT_B8G8R8A8_UNORM;
        srvd.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvd.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(upscaledTex, &srvd,
                                          srv_.GetAddressOf());
        lastTex = upscaledTex;
    }

    ctx_->OMSetRenderTargets(1, rtv_.GetAddressOf(), nullptr);
    ctx_->PSSetShaderResources(0, 1, srv_.GetAddressOf());

    // Draw fullscreen quad (4 vertices, no vertex buffer — VS uses SV_VertexID)
    ctx_->Draw(4, 0);

    // Present without vsync (0) for minimum latency; change to 1 for vsync
    swapChain_->Present(0, 0);
}

bool DisplayEngine::PumpMessages() {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return false;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return true;
}

void DisplayEngine::Shutdown() {
    if (swapChain_) swapChain_->SetFullscreenState(FALSE, nullptr);
    srv_.Reset();
    sampler_.Reset();
    rtv_.Reset();
    ps_.Reset();
    vs_.Reset();
    swapChain_.Reset();
    ctx_.Reset();
    device_.Reset();
    if (hwnd_) { DestroyWindow(hwnd_); hwnd_ = nullptr; }
}

DisplayEngine::~DisplayEngine() { Shutdown(); }
