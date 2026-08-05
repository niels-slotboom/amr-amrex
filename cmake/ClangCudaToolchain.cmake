# Set compilers BEFORE project() is called
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_LINKER clang++)

# Instruct CMake to use clang++ for CUDA host & device passes
set(CMAKE_CUDA_COMPILER clang++)
set(CMAKE_CUDA_HOST_COMPILER clang++)

# Path to CUDA SDK
set(CUDA_PATH "/opt/cuda" CACHE PATH "Path to CUDA SDK")

# Pass standard & architecture directly to Clang
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ADJUST THIS FOR DIFFERENT ARCHITECTURE
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to target")

# Add to your CXX and CUDA flags init
set(CMAKE_CXX_FLAGS_INIT "-Wno-deprecated-attributes -Wno-unknown-cuda-version")
set(CMAKE_CUDA_FLAGS_INIT "--cuda-path=${CUDA_PATH} --cuda-gpu-arch=sm_86 -std=c++20 -Wno-deprecated-attributes -Wno-unknown-cuda-version")

# Emit compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands" FORCE)