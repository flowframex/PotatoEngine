cmake_minimum_required(VERSION 3.10)
project(PotatoEngine)

# ─────────────────────────────────────────────────────────────────────────────
#  PotatoEngine — CMakeLists.txt
#
#  Uses the LEGACY FindCUDA module (not CUDAToolkit).
#  This is required because CUDA 8.0 — the last version supporting
#  Fermi (sm_21 / GT730 GF108) — predates the modern CMake CUDA support.
#
#  CUDA 9+ dropped Fermi. CMake will error if you try to use a newer CUDA.
# ─────────────────────────────────────────────────────────────────────────────

enable_language(CXX)

if(NOT WIN32)
    message(FATAL_ERROR "PotatoEngine requires Windows (DXGI / D3D11)")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Legacy FindCUDA ──────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)

message(STATUS "CUDA version : ${CUDA_VERSION_STRING}")
message(STATUS "CUDA toolkit : ${CUDA_TOOLKIT_ROOT_DIR}")

if(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
    message(FATAL_ERROR
        "\n"
        "  *** WRONG CUDA VERSION ***\n"
        "  CUDA ${CUDA_VERSION} dropped Fermi (sm_21) support.\n"
        "  Your GT730 GF108 needs CUDA 8.0.\n"
        "  Get it: https://developer.nvidia.com/cuda-80-download-archive\n"
        "  Then: cmake -DCUDA_TOOLKIT_ROOT_DIR=\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0\" ..\n")
endif()

# ─── NVCC flags for Fermi sm_21 ───────────────────────────────────────────────
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_20,code=sm_21
    -gencode arch=compute_20,code=sm_20
    --use_fast_math
    -O3
    -Xptxas=-v
    -Xcompiler=/O2,/W3,/EHsc,/wd4819
)

set(CUDA_SEPARABLE_COMPILATION ON)

# ─── Sources ──────────────────────────────────────────────────────────────────
cuda_add_executable(PotatoEngine WIN32
    src/main.cpp
    src/CaptureEngine.cpp
    src/DisplayEngine.cpp
    src/UpscaleEngine.cu
)

target_include_directories(PotatoEngine PRIVATE
    src/
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(PotatoEngine
    ${CUDA_LIBRARIES}
    d3d11.lib
    dxgi.lib
    d3dcompiler.lib
    dxguid.lib
)

if(MSVC)
    target_compile_options(PotatoEngine PRIVATE /MP /EHsc)
    set_property(TARGET PotatoEngine PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ─── Copy cudart DLL to output ────────────────────────────────────────────────
add_custom_command(TARGET PotatoEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_80.dll"
        "$<TARGET_FILE_DIR:PotatoEngine>"
    COMMENT "Copying cudart64_80.dll"
)

install(TARGETS PotatoEngine DESTINATION bin)
install(FILES "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_80.dll" DESTINATION bin)

message(STATUS "")
message(STATUS "  PotatoEngine ready  |  Target: GT730 GF108 (sm_21)  |  CUDA ${CUDA_VERSION_STRING}")
message(STATUS "")
