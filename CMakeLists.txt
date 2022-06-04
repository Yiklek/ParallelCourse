cmake_minimum_required(VERSION 3.16)

project(gemm)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)
#find_package(MPI)
find_package(BLAS REQUIRED)


message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
message(STATUS "Using BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
message(STATUS "Using BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

if(NOT BLAS_INCLUDE_DIRS)
    message(FATAL_ERROR "BLAS_INCLUDE_DIRS must be set")
endif()
message(STATUS "Using BLAS_INCLUDE_DIRS: ${BLAS_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}) 
include_directories(${BLAS_INCLUDE_DIRS})

set(GEMM_LINK_LIBS ${BLAS_LIBRARIES})

set(SOURCE_FILES gemm.cc)
add_executable(gemm ${SOURCE_FILES})
#set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CUDA_ENABLE)
    enable_language(CUDA)
    target_compile_definitions(gemm PRIVATE CUDA_ENABLE)
    add_library(cugemm gemm.cu)
    target_compile_features(cugemm PUBLIC cxx_std_11)
    target_link_libraries(cugemm -lcublas)
    set_target_properties(cugemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND GEMM_LINK_LIBS cugemm)
endif()

target_link_libraries(gemm ${GEMM_LINK_LIBS})
