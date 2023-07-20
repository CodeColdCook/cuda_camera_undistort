cmake_minimum_required(VERSION 3.10)
project(cuda_camera_undistort 
  VERSION 1.1
  LANGUAGES CXX)

#准备的版本设置
SET(PROJECT_VERSION_MAJOR 1)
SET(PROJECT_VERSION_MINOR 0)
SET(PROJECT_VERSION_PATCH 1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings")	

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV 4.6 REQUIRED)

find_package(CUDA)
include_directories( ${OpenCV_INCLUDE_DIRS})
message("-- CUDA version: ${CUDA_VERSION}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_53 SM_62")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62)

  if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_72")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_72,code=sm_72)
  endif()

  if(CUDA_VERSION_MAJOR GREATER 10)
    message("-- CUDA ${CUDA_VERSION} detected (${CMAKE_SYSTEM_PROCESSOR}), enabling SM_87")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_87,code=sm_87)
  endif()
endif()

if(NOT DEFINED MAP_X_PATH) 
  set(MAP_X_PATH "/opt/pmtd/config/remap_config/map_X.yaml")
  message(STATUS "${PROJECT_NAME}: set MAP_X_PATH to ${MAP_X_PATH}" )
endif()

if(NOT DEFINED MAP_Y_PATH) 
  set(MAP_Y_PATH "/opt/pmtd/config/remap_config/map_X.yaml")
  message(STATUS "${PROJECT_NAME}: set MAP_Y_PATH to ${MAP_Y_PATH}" )
endif()

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# pkgconfig
write_basic_package_version_file(
  ${version_config}
  VERSION ${CUDA_CAMERA_UNDISTORT_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/config.h.in
  ${PROJECT_SOURCE_DIR}/config.h
  @ONLY
)

cuda_add_library(${PROJECT_NAME} SHARED main.cu)
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  -lGLESv2
  -lEGL
  -lcuda
  -lrt
  )
