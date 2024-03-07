# This CMake config hopefully makes it easier to compile.
# Ensure the CUDA Toolkit is available on your path. Then run:
#   For  GCC: `cmake -B build . && cmake --build build`
#   For MSVC: `cmake -B build . && cmake --build build --config Release`
# You can also use the following options and variables
#  - COMPUTE_BACKEND: Set to `cpu`, `cuda`, or `mps` to select the backend
#  - NO_CUBLASLT: Default OFF, will skip building/linking CUBLASLT support
#  - CUDA_VERSION: The expected CUDA version, for sanity checking. The actual version
#                  is whatever CMake finds on your path.
#  - COMPUTE_CAPABILITY: Which GPU Arch/Compute codes to provide to NVCC.
#                        Separate by semicolons, i.e. `-DCOMPUTE_CAPABILITY=89;90`
#                        Check your compute capability here: https://developer.nvidia.com/cuda-gpus
#  - PTXAS_VERBOSE: Pass the `-v` option to the PTX Assembler
cmake_minimum_required(VERSION 3.22.1)

project(bitsandbytes LANGUAGES CXX)

# Define included source files
set(CPP_FILES csrc/common.cpp csrc/cpu_ops.cpp csrc/pythonInterface.cpp)
set(CUDA_FILES csrc/ops.cu csrc/kernels.cu)
set(MPS_FILES csrc/mps_ops.mm)
set(METAL_FILES csrc/mps_kernels.metal)
# C++ sources are always included
list(APPEND SRC_FILES ${CPP_FILES})

set(COMPUTE_BACKEND "hip" CACHE STRING "The compute backend to use (cpu, cuda, mps, hip)")
set_property(CACHE COMPUTE_BACKEND PROPERTY STRINGS cpu cuda mps hip)
option(PTXAS_VERBOSE "Pass through -v flag to PTX Assembler" OFF)

##########################################3

#this method follows "Consuming the HIP API in C++ code"
#https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html#consuming-the-hip-api-in-c-code

message("Is it Window?: " ${CMAKE_HOST_WIN32})
message("Is it Linux?: " ${CMAKE_HOST_UNIX})

# Find OpenMP.
#find_package(OpenMP REQUIRED)





# Source files.
#set(CPP_SOURCES ${CMAKE_SOURCE_DIR}/common.cpp ${CMAKE_SOURCE_DIR}/common.h ${CMAKE_SOURCE_DIR}/hip-stream.cpp)
#message("CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})

# Preparing the executable.
#Add an executable target called "test_openmp_helloworld" to be built from the source files listed in the command invocation.
#add_executable(test_add4 ${CPP_SOURCES})


#####################################

set(BNB_OUTPUT_NAME "bitsandbytes")

message(STATUS "Configuring ${PROJECT_NAME} (Backend: ${COMPUTE_BACKEND})")

set(BUILD_HIP ON)

if(BUILD_HIP)
    #enable_language(CUDA) # This will fail if CUDA is not found
    #find_package(CUDAToolkit REQUIRED)
    set(CMAKE_BUILD_TYPE Release)
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm-6.0.2 /opt/rocm)
    set(CMAKE_HIP_PLATFORM amd)
    # Find HIP.
    # The user may override AMDGPU_TARGETS defined in the HIP config file
    # to select the AMDGPU archs to compile for.
    # ex. set(AMDGPU_TARGETS "gfx803;gfx900;gfx906")
    find_package(hip REQUIRED)

    # Link Libraries - HIP Device and OpenMP.
    #target_compile_options(test_vector_add PRIVATE ${OpenMP_CXX_FLAGS})
    #target_link_libraries(test_vector_add PRIVATE hip::device ${OpenMP_CXX_FLAGS})
    # Set compiler and linker.
    if(NOT WIN32)
      set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
      set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
    endif()
    message("Current CMAKE_CXX_COMPILER (should show hipcc): " ${CMAKE_CXX_COMPILER})
    message("Current CMAKE_CXX_LINKER (should show hipcc): " ${CMAKE_CXX_LINKER})

    # Expose a cache variable that the user can set to ensure the correct version of CUDA is found
    set(ROCM_VERSION "602" CACHE STRING "Expected ROCM Version Shortcode")
    message(STATUS "ROCM Version: ${ROCM_VERSION_SHORT} ")
    
    message(STATUS "CUDA Targets: ${CMAKE_HIP_ARCHITECTURES}")
    #message(STATUS "CUDA NVCC Flags: ${CMAKE_CUDA_FLAGS}")

    list(APPEND SRC_FILES ${CUDA_FILES})

    string(APPEND BNB_OUTPUT_NAME "_rocm${ROCM_VERSION_SHORT}")
    if(NO_CUBLASLT)
        string(APPEND BNB_OUTPUT_NAME "_nocublaslt")
    endif()
    add_compile_definitions(BUILD_HIP)
else()
    string(APPEND BNB_OUTPUT_NAME "_cpu")
    message("Current CMAKE_CXX_COMPILER (should show hipcc): " ${CMAKE_CXX_COMPILER})
    message("Current CMAKE_CXX_LINKER (should show hipcc): " ${CMAKE_CXX_LINKER})
    set(GPU_SOURCES)
endif()

set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CXX)
#either add_library or add_executable must exist
add_library(bitsandbytes SHARED ${SRC_FILES})
#target here must have been created by either add_executable or add_library
target_compile_features(bitsandbytes PUBLIC cxx_std_11)
target_include_directories(bitsandbytes PUBLIC csrc include)


if(BUILD_HIP)
  #list(APPEND CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH})
  message("CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
  find_package(hipblas REQUIRED)
  find_package(hipsparse REQUIRED)
  target_include_directories(bitsandbytes PUBLIC ${CMAKE_PREFIX_PATH})
  target_link_libraries(bitsandbytes PRIVATE roc::hipblas roc::hipsparse)
  if(NO_CUBLASLT)
      target_compile_definitions(bitsandbytes PUBLIC NO_CUBLASLT)
  else()
      target_link_libraries(bitsandbytes PRIVATE hip::host)
  endif()
  set_target_properties(bitsandbytes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

set_target_properties(bitsandbytes PROPERTIES OUTPUT_NAME ${BNB_OUTPUT_NAME})

set_target_properties(bitsandbytes PROPERTIES LIBRARY_OUTPUT_DIRECTORY bitsandbytes)
