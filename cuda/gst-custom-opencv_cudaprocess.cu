/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cudaEGL.h"

#define BOX_W 128
#define BOX_H 128
#define CORD_X 64
#define CORD_Y 64
#define MAX_BUFFERS 30
// static BBOX rect_data[MAX_BUFFERS];
#define USE_OPENCV 1

#if USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
cv::cuda::GpuMat gpumap_x_, gpumap_y_;
std::string map_path_ = "/opt/pmtd/config/lidar_camera_config/pmtd104";

#endif

#if defined(__cplusplus)
extern "C" void Handle_EGLImage(EGLImageKHR image);
extern "C" {
#endif

__device__ __host__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#if USE_OPENCV

// #include <opencv2/cudafilters.hpp>
// #include <opencv2/cudaimgproc.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/opencv.hpp>

// #include "cudaABGR.h"
// #include "imageIO.h"

__global__ void abgr2bgra_kernel(cv::cuda::PtrStepSz<uchar4> src,
                                 cv::cuda::PtrStepSz<uchar4> dst) {
  int i = threadIdx.x +
          blockIdx.x * blockDim.x;  // thread在x方向的全局索引，也就是列坐标
  int j = threadIdx.y +
          blockIdx.y * blockDim.y;  // thread在y方向的全局索引，也就是行坐标
  // if (i == 0 && j == 0)
  //   printf("Grid size: (%d, %d)\n", gridDim.x, gridDim.y);  //
  //   可用printf来debug
  if (j < src.rows && i < src.cols)  //! 要判断是否越界！！！
  {
    uchar4 px = src(j, i);
    dst(j, i) = make_uchar4(px.y, px.z, px.w, px.x);
  }
}

#if 0
// 测试完成，可以使用，但去畸变以及该过程整体耗时约23ms
static void cv_process(void *pdata, int32_t width, int32_t height) {
  cv::cuda::GpuMat abgr_mat(height, width, CV_8UC4, pdata);
  static int abgr2bgra_order[] = {1, 2, 3, 0};
  static int bgra2abgr_order[] = {3, 0, 1, 2};
  cv::cuda::swapChannels(abgr_mat, abgr2bgra_order);
  cv::cuda::GpuMat bgra_mat(1920, 1080, CV_8UC4);
  cv::cuda::remap(abgr_mat, bgra_mat, gpumap_x_, gpumap_y_, cv::INTER_LINEAR);
  cv::cuda::swapChannels(bgra_mat, bgra2abgr_order);
  bgra_mat.copyTo(abgr_mat);

  // gpu_mat_src.download(image_dst);
  // cv::imwrite("./swap_back_result.jpg", image_dst);
}
#endif

#if 1
// 测试完成，可以使用，去畸变以及该过程整体耗时约16-17ms
static void cv_process(void *pdata, int32_t width, int32_t height) {
  cv::cuda::GpuMat abgr_mat(height, width, CV_8UC4, pdata);
  cv::cuda::GpuMat bgra_mat(1920, 1080, CV_8UC4);
  cv::cuda::remap(abgr_mat, bgra_mat, gpumap_x_, gpumap_y_, cv::INTER_LINEAR);
  bgra_mat.copyTo(abgr_mat);
}
#endif

#if 0
// 测试完成，无法使用，去畸变以及该过程整体耗时约15-16ms
static void cv_process(void *pdata, int32_t width, int32_t height) {
  cv::cuda::GpuMat abgr_mat(height, width, CV_8UC4, pdata);
  cv::cuda::remap(abgr_mat, abgr_mat, gpumap_x_, gpumap_y_, cv::INTER_LINEAR);
}
#endif

#if 0
// 测试完成，可以使用，该过程整体耗时约7ms
static void cv_process(void *pdata, int32_t width, int32_t height) {
// do nothing
}
#endif

#endif

__global__ void addLabelsKernel(int *pDevPtr, int pitch) {
  int row = blockIdx.y * blockDim.y + threadIdx.y + BOX_H;
  int col = blockIdx.x * blockDim.x + threadIdx.x + BOX_W;
  char *pElement = (char *)pDevPtr + row * pitch + col * 4;
  // char *pElement = (char *)pDevPtr + row * pitch + col * 2;
  pElement[1] = 255;
  pElement[2] = 255;
  pElement[3] = 255;
  return;
}

static int addLabels(CUdeviceptr pDevPtr, int pitch) {
  dim3 threadsPerBlock(8, 8, 1);
  dim3 blocks(iDivUp(BOX_W, threadsPerBlock.x),
              iDivUp(BOX_H, threadsPerBlock.y), 1);
  addLabelsKernel<<<blocks, threadsPerBlock>>>((int *)pDevPtr, pitch);
  return 0;
}

typedef enum {
  COLOR_FORMAT_Y8 = 0,
  COLOR_FORMAT_U8_V8,
  COLOR_FORMAT_RGBA,
  COLOR_FORMAT_NONE
} ColorFormat;

typedef struct {
  /**
   * cuda-process API
   *
   * @param image   : EGL Image to process
   * @param userPtr : point to user alloc data, should be free by user
   */
  void (*fGPUProcess)(EGLImageKHR image, void **userPtr);

  /**
   * pre-process API
   *
   * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
   * @param smemsize   : surfaces size array
   * @param swidth     : surfaces width array
   * @param sheight    : surfaces height array
   * @param spitch     : surfaces pitch array
   * @param sformat    : surfaces format array
   * @param nsurfcount : surfaces count
   * @param userPtr    : point to user alloc data, should be free by user
   */
  void (*fPreProcess)(void **sBaseAddr, unsigned int *smemsize,
                      unsigned int *swidth, unsigned int *sheight,
                      unsigned int *spitch, ColorFormat *sformat,
                      unsigned int nsurfcount, void **userPtr);

  /**
   * post-process API
   *
   * @param sBaseAddr  : Mapped Surfaces(YUV) pointers
   * @param smemsize   : surfaces size array
   * @param swidth     : surfaces width array
   * @param sheight    : surfaces height array
   * @param spitch     : surfaces pitch array
   * @param sformat    : surfaces format array
   * @param nsurfcount : surfaces count
   * @param userPtr    : point to user alloc data, should be free by user
   */
  void (*fPostProcess)(void **sBaseAddr, unsigned int *smemsize,
                       unsigned int *swidth, unsigned int *sheight,
                       unsigned int *spitch, ColorFormat *sformat,
                       unsigned int nsurfcount, void **userPtr);
} CustomerFunction;

void init(CustomerFunction *pFuncs);

#if defined(__cplusplus)
}
#endif

/**
 * Dummy custom pre-process API implematation.
 * It just access mapped surface userspace pointer &
 * memset with specific pattern modifying pixel-data in-place.
 *
 * @param sBaseAddr  : Mapped Surfaces pointers
 * @param smemsize   : surfaces size array
 * @param swidth     : surfaces width array
 * @param sheight    : surfaces height array
 * @param spitch     : surfaces pitch array
 * @param nsurfcount : surfaces count
 */
static void pre_process(void **sBaseAddr, unsigned int *smemsize,
                        unsigned int *swidth, unsigned int *sheight,
                        unsigned int *spitch, ColorFormat *sformat,
                        unsigned int nsurfcount, void **usrptr) {
  // int x, y;
  // char *uv = NULL;
  // unsigned char *rgba = NULL;
  // if (sformat[1] == COLOR_FORMAT_U8_V8) {
  //   uv = (char *)sBaseAddr[1];
  //   for (y = 0; y < BOX_H; ++y) {
  //     for (x = 0; x < BOX_W; ++x) {
  //       uv[y * spitch[1] + 2 * x] = 0;
  //       uv[y * spitch[1] + 2 * x + 1] = 0;
  //     }
  //   }
  // } else if (sformat[0] == COLOR_FORMAT_RGBA) {
  //   rgba = (unsigned char *)sBaseAddr[0];
  //   for (y = 0; y < BOX_H * 2; y++) {
  //     for (x = 0; x < BOX_W * 8; x += 4) {
  //       rgba[x + 0] = 0;
  //       rgba[x + 1] = 0;
  //       rgba[x + 2] = 0;
  //       rgba[x + 3] = 0;
  //     }
  //     rgba += spitch[0];
  //   }
  // }
  /* add your custom pre-process here
     we draw a green block for demo */
}

/**
 * Dummy custom post-process API implematation.
 * It just access mapped surface userspace pointer &
 * memset with specific pattern modifying pixel-data in-place.
 *
 * @param sBaseAddr  : Mapped Surfaces pointers
 * @param smemsize   : surfaces size array
 * @param swidth     : surfaces width array
 * @param sheight    : surfaces height array
 * @param spitch     : surfaces pitch array
 * @param nsurfcount : surfaces count
 */
static void post_process(void **sBaseAddr, unsigned int *smemsize,
                         unsigned int *swidth, unsigned int *sheight,
                         unsigned int *spitch, ColorFormat *sformat,
                         unsigned int nsurfcount, void **usrptr) {
  /* add your custom post-process here
     we draw a green block for demo */
  // int x, y;
  // char *uv = NULL;
  // int xoffset = (CORD_X * 4);
  // int yoffset = (CORD_Y * 2);
  // unsigned char *rgba = NULL;
  // if (sformat[1] == COLOR_FORMAT_U8_V8) {
  //   uv = (char *)sBaseAddr[1];
  //   for (y = 0; y < BOX_H; ++y) {
  //     for (x = 0; x < BOX_W; ++x) {
  //       uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2)] = 0;
  //       uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2) + 1] = 0;
  //     }
  //   }
  // } else if (sformat[0] == COLOR_FORMAT_RGBA) {
  //   rgba = (unsigned char *)sBaseAddr[0];
  //   rgba += ((spitch[0] * yoffset) + xoffset);
  //   for (y = 0; y < BOX_H * 2; y++) {
  //     for (x = 0; x < BOX_W * 8; x += 4) {
  //       rgba[(x + xoffset) + 0] = 0;
  //       rgba[(x + xoffset) + 1] = 0;
  //       rgba[(x + xoffset) + 2] = 0;
  //       rgba[(x + xoffset) + 3] = 0;
  //     }
  //     rgba += spitch[0];
  //   }
  // }
}

/* This filter will be created by init() function below */
// cv::Ptr<cv::cuda::Filter> filter;

/* pdata is ABGR */

/**
 * Performs CUDA Operations on egl image.
 *
 * @param image : EGL image
 */
static void gpu_process(EGLImageKHR image, void **usrptr) {
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;

  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(&pResource, image,
                                      CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    return;
  }

  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsSubResourceGetMappedArray failed\n");
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf("cuCtxSynchronize failed \n");
  }

  if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_RGBA) {
      /* Perform now your custom opencv processing */
      // printf("pitch %d\n", eglFrame.pitch);
#if USE_OPENCV
      cv_process(eglFrame.frame.pPitch[0], eglFrame.width, eglFrame.height);
#else
      addLabels((CUdeviceptr)eglFrame.frame.pPitch[0], eglFrame.pitch);
#endif

    } else if (eglFrame.eglColorFormat ==
               CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR) {
      /* Rectangle label in plan UV , you can replace this with any cuda
       * algorithms */
// #if USE_OPENCV
//       cv_process(eglFrame.frame.pPitch[1], eglFrame.width, eglFrame.height);
// #else
//       addLabels((CUdeviceptr)eglFrame.frame.pPitch[0], eglFrame.pitch);
// #endif
    } else {
      printf("Invalid eglcolorformat for opencv %d\n", eglFrame.eglColorFormat);
    }
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf("cuCtxSynchronize failed after memcpy \n");
  }

  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
  }
}

extern "C" void init(CustomerFunction *pFuncs) {
  // load remap map_x & map_y
  cv::Mat map_x, map_y;
  cv::FileStorage fs(map_path_ + "/map_X.yaml", cv::FileStorage::READ);
  fs["map_X"] >> map_x;
  fs.release();
  fs.open(map_path_ + "/map_Y.yaml", cv::FileStorage::READ);
  fs["map_Y"] >> map_y;
  fs.release();
  gpumap_x_.upload(map_x);
  gpumap_y_.upload(map_y);

  // cv::Mat image_src;
  // image_src = cv::imread("/home/nvidia/bgra_mat.jpg");
  // cv::cuda::GpuMat gpu_mat_src(1920, 1080, CV_8UC3);
  // gpu_mat_src.upload(image_src);
  // // cv::cuda::GpuMat abgr_mat(1920, 1080, CV_8UC3);
  // // cv::cuda::cvtColor(gpu_mat_src, abgr_mat, cv::COLOR_BGRA2BGR);

  // cv::Mat image_dst;
  // gpu_mat_src.download(image_dst);
  // cv::imwrite("./swap_result.jpg", image_dst);
  // cv::cuda::GpuMat gpu_mat_dst(1920, 1080, CV_8UC3);
  // cv::cuda::remap(gpu_mat_src, gpu_mat_dst, gpumap_x_, gpumap_y_,
  // cv::INTER_NEAREST); gpu_mat_dst.download(image_dst);
  // cv::imwrite("./remap_result.jpg", image_dst);

  pFuncs->fPreProcess = nullptr;
  pFuncs->fGPUProcess = nullptr;
  pFuncs->fPostProcess = nullptr;
}