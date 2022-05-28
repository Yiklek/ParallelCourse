cmake_minimum_required(VERSION 3.16)

project(gemm)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)
find_package(MPI)
find_package(BLAS REQUIRED)

set(SOURCE_FILES gemm.cc)

message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")
message(STATUS "Using BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
message(STATUS "Using BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

if(NOT BLAS_INCLUDE_DIRS)
    message(FATAL_ERROR "BLAS_INCLUDE_DIRS must be set")
endif()
message(STATUS "Using BLAS_INCLUDE_DIRS: ${BLAS_INCLUDE_DIRS}")
include_directories(${BLAS_INCLUDE_DIRS}) 
add_executable(gemm ${SOURCE_FILES})
target_link_libraries(gemm BLAS::BLAS)
