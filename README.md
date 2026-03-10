# 🥔 PotatoEngine — GTXxx Edition

**Zero-copy CUDA upscaler: 540p → 1080p, entirely in VRAM.**  
Built for NVIDIA GT730 **GF108 Fermi (sm_21)** — the hardest GPU to support in 2024.  
No RAM↔VRAM transfers. No input lag. Pure Lanczos-2 + sharpening on the GPU.

---

## ⚠️ CUDA Version Reality Check (Read This First)

| CUDA Version | Fermi sm_21 support | Kepler sm_35 support |
|---|---|---|
| CUDA 7.5 | ✅ Yes | ✅ Yes |
| **CUDA 8.0** | ✅ **Yes — last version** | ✅ Yes |
| CUDA 9.0+ | ❌ **Dropped Fermi** | ✅ Yes |
| CUDA 12.x | ❌ Dropped Fermi | ✅ Yes |

**CUDA 8.0 is the ONLY version that compiles sm_21 for your GT730 GF108.**  
The GitHub Actions workflow in this repo downloads and installs CUDA 8.0 automatically.

---

## Verify Your GT730 Variant

There are **two completely different GT730 chips:**

```cmd
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

| Output | Chip | You need |
|--------|------|----------|
| `compute_cap = 2.1` | **GF108 — Fermi** | This repo (CUDA 8 build) |
| `compute_cap = 3.5` | GK208 — Kepler | Rebuild with CUDA 12 |

If you're not sure, open **GPU-Z** → look at the `Revision` field:
- `GF108` = Fermi → compute 2.1 → **you're in the right place**
- `GK208` = Kepler → compute 3.5 → use the CUDA 12 workflow instead

---

## How the Pipeline Works (Zero RAM↔VRAM)

```
Game renders at 540p (960×540)
        │
  DXGI Desktop Duplication
        │
  D3D11 Texture  ←────────────────────── stays in VRAM (4GB DDR3)
        │
  cudaGraphicsD3D11RegisterResource      ← CUDA maps the D3D texture directly
        │                                   NO cudaMemcpy, NO RAM transfer
  surf2Dread (CUDA reads source pixels)
        │
  ┌─────────────────────────────────┐
  │  Kernel 1: Lanczos-2 Upscale   │  960×540 → 1920×1080
  │  Kernel 2: Laplacian Sharpen   │  adaptive unsharp mask
  │  Kernel 3: Saturation boost    │  +5% colour, optional
  └─────────────────────────────────┘
        │
  surf2Dwrite (CUDA writes output pixels)
        │
  D3D11 SRV → Fullscreen Quad → SwapChain::Present
        │
  1080p on your monitor
```

The captured frame and the upscaled frame **never leave VRAM**.  
CUDA accesses the D3D11 texture via `surf2Dread`/`surf2Dwrite` — this is the  
`cudaGraphicsD3D11RegisterResource` interop path, zero-copy by design.

---

## Algorithm: Lanczos-2

Each output pixel is computed from a **4×4 neighbourhood** in the source:

```
weight(x) = sinc(x) × sinc(x/2),  |x| < 2
          = 0                       |x| ≥ 2
```

Then sharpening via Laplacian:
```
out[p] = (1 + k) × center − (k/4) × (N + S + E + W)
```
Default `k = 0.45`. Tune live with `Alt+F10` / `Alt+F11`.

---

## Fermi-Specific Optimisations in the Code

| Problem | Solution |
|---------|----------|
| Fermi max 63 registers/thread | `__launch_bounds__(128, 2)` on Lanczos kernel |
| High register pressure → spill | Loop instead of `#pragma unroll` in Lanczos |
| No `__ldg()` (sm_30+) | Not used (surface objects used instead) |
| No warp shuffles (sm_30+) | Not used |
| 4×4 Lanczos too heavy at 16×16? | Reduced to **8×16 block** for heavy kernel |

---

## Build via GitHub Actions (Recommended — No Local Setup Needed)

1. Fork this repo on GitHub
2. Push any commit  
   → Actions tab → **"Build PotatoEngine (Fermi sm_21)"** workflow triggers  
   → It downloads CUDA 8.0 (~1.4 GB), installs it, compiles
3. When it finishes: **Actions → latest run → Artifacts → download**  
   `PotatoEngine-Fermi-sm21-CUDA8-windows-x64.zip`
4. Extract → run `PotatoEngine.exe` as Administrator

To make a GitHub Release:
```bash
git tag v1.0.0
git push origin v1.0.0
```
The workflow attaches the `.exe` and `cudart64_80.dll` to the release automatically.

---

## Build Locally (Requires CUDA 8.0 Installed)

### Step 1: Get CUDA 8.0
Download from NVIDIA's legacy archive:  
https://developer.nvidia.com/cuda-80-download-archive

Install it. Default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`

### Step 2: Get VS 2015 Build Tools
CUDA 8.0 supports up to VS 2015 as host compiler.  
Download: https://aka.ms/vs/15/release/vs_buildtools.exe  
Install the **C++ build tools** component (v140 toolset).

### Step 3: Configure and Build
```cmd
git clone https://github.com/YOUR_USERNAME/PotatoEngine
cd PotatoEngine

cmake -B build ^
  -G "Visual Studio 15 2017" ^
  -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0"

cmake --build build --config Release
```

Binary at: `build\Release\PotatoEngine.exe`

---

## Usage

1. Set your game resolution to **960×540** (windowed or borderless windowed)
2. Run `PotatoEngine.exe` **as Administrator**  
   (DXGI Desktop Duplication requires admin rights on most Win10 configs)
3. A black 1920×1080 fullscreen window appears — engine is running and capturing
4. Play your game — upscaled 1080p output shows on screen

### Hotkeys

| Hotkey | Action |
|--------|--------|
| `Alt + F9`  | Toggle overlay on/off |
| `Alt + F10` | Sharpness +0.1 |
| `Alt + F11` | Sharpness −0.1 |
| `Escape`    | Quit |

### Sharpness Tuning Guide

| Value | Result |
|-------|--------|
| 0.0 | Pure Lanczos — smooth |
| 0.3 | Mild — good for anime / cartoon games |
| **0.45** | **Default — natural, sharp** |
| 0.6 | Crisp — good for pixel art / old games |
| 0.8 | Aggressive — some halos on edges |

---

## Expected Performance on GT730 Fermi

Fermi has fewer CUDA cores and slower memory than Kepler, but Lanczos-2  
is a pure floating-point workload with no tensor/RT core requirements:

| Resolution | Approx. FPS |
|------------|-------------|
| 540p → 1080p | ~25–45 fps |
| 480p → 1080p | ~30–55 fps |

If your game runs at 30 fps, the upscale overhead is well within budget.  
The engine runs asynchronously — it captures and upscales independently  
of your game's frame rate.

---

## Troubleshooting

**"CUDA 9.x dropped Fermi" error during build**  
→ CUDA_TOOLKIT_ROOT_DIR is pointing to the wrong CUDA version  
→ Set it explicitly: `-DCUDA_TOOLKIT_ROOT_DIR="C:\...\CUDA\v8.0"`

**"DuplicateOutput failed: 0x887A0004"**  
→ Run as Administrator  
→ Disable HDR on your display if enabled

**"No CUDA devices found"**  
→ Update NVIDIA driver for GT730 (use the latest legacy driver for Fermi)  
→ Run `nvidia-smi` in CMD — if it fails, reinstall drivers

**Black screen / nothing visible**  
→ Press `Alt+F9` to toggle the overlay  
→ Make sure the game is on the primary monitor

**Slow / stuttering / poor FPS**  
→ Close Chrome (it consumes VRAM even with your 4GB)  
→ Check GPU temp with GPU-Z: Fermi throttles around 95°C  
→ Lower sharpness (`Alt+F11`) — the sharpen kernel is the most expensive

**"illegal instruction" crash**  
→ Wrong build — you have sm_21 but the binary was compiled for sm_35+  
→ Use the Fermi build from this repo (CUDA 8, sm_21 binary)

---

## Project Structure

```
PotatoEngine/
├── .github/workflows/build.yml   ← CI: downloads CUDA 8, builds for sm_21
├── CMakeLists.txt                ← Legacy FindCUDA (for CUDA 8 compat)
├── README.md
└── src/
    ├── main.cpp                  ← Entry point, pipeline loop, hotkeys
    ├── CaptureEngine.h/.cpp      ← DXGI Desktop Duplication (VRAM capture)
    ├── DisplayEngine.h/.cpp      ← D3D11 fullscreen output window
    ├── UpscaleEngine.cuh         ← CUDA engine interface
    └── UpscaleEngine.cu          ← Lanczos-2, sharpen, saturation kernels
                                     (Fermi-optimised: __launch_bounds__,
                                      manual loops, 8×16 block for Lanczos)
```

---

## License

MIT.

You've squeezed a Fermi GPU past its official software support date by 8+ years.  
You deserve an upscaler to match that energy.
