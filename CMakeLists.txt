cmake_minimum_required(VERSION 3.18)

project(fdtdz_jax LANGUAGES CXX CUDA)

# To run cmake independently of "pip3 install", install pybind11 via
# `pip3 install "pybind11[global]"`, see 
# https://pybind11.readthedocs.io/en/stable/installing.html#include-with-pypi
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Needed to link against CUDA runtime/driver APIs.
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Needed for reduced precision.

include_directories(${CMAKE_CURRENT_LIST_DIR}/cuda)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
  gpu_ops
  ${CMAKE_CURRENT_LIST_DIR}/cuda/kernel_jax.cc.cu
  ${CMAKE_CURRENT_LIST_DIR}/cuda/jax_ops.cc)
target_link_libraries(gpu_ops PUBLIC CUDA::cudart CUDA::cuda_driver)
install(TARGETS gpu_ops DESTINATION fdtdz_jax)

