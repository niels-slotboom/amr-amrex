cmake_minimum_required(VERSION 3.28)

project(AmrAMReX VERSION 0.1.0 LANGUAGES CXX CUDA)

# CRITICAL: Prevent CMake from injecting fatbinary rules globally
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_BUILD_TYPE Release)

# Global C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Architecture Configuration
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "" FORCE)
set(AMReX_CUDA_ARCH 8.9 CACHE STRING "" FORCE)

# AMReX Configuration Options
set(AMReX_SPACEDIM 3 CACHE STRING "AMReX Space Dimension" FORCE)
set(AMReX_CUDA ON CACHE BOOL "Enable CUDA in AMReX" FORCE)
set(AMReX_GPU_BACKEND CUDA CACHE STRING "AMReX GPU Backend" FORCE)
set(AMReX_ENABLE_TESTS OFF CACHE BOOL "Disable AMReX Tests" FORCE)
set(AMReX_CUDA_RDC OFF CACHE BOOL "Disable RDC for pure Clang toolchain" FORCE)
set(AMReX_SETUP_COMPILER_FLAGS OFF CACHE BOOL "Disable internal AMReX compiler flag setup" FORCE)

# Fetch AMReX
include(FetchContent)
FetchContent_Declare(
    amrex
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    GIT_TAG        development
)

FetchContent_MakeAvailable(amrex)

add_subdirectory(src/boxer-test)