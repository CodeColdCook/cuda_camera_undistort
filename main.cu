#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core/persistence.hpp>
#include <opencv2/cudawarping.hpp>

#include "config.h"
#include "cudaEGL.h"

cv::cuda::GpuMat gpumap_x_, gpumap_y_;
bool initialized = false;

#if defined(__cplusplus)
extern "C" void Handle_EGLImage(EGLImageKHR image);
extern "C" {
#endif

__device__ __host__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
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

static void cv_process(void *pdata, int32_t width, int32_t height) {
  cv::cuda::GpuMat abgr_mat(height, width, CV_8UC4, pdata);
  cv::cuda::GpuMat bgra_mat(1920, 1080, CV_8UC4);
  cv::cuda::remap(abgr_mat, bgra_mat, gpumap_x_, gpumap_y_, cv::INTER_LINEAR);
  bgra_mat.copyTo(abgr_mat);
}

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
      if (initialized) {
        cv_process(eglFrame.frame.pPitch[0], eglFrame.width, eglFrame.height);
      } else {
        printf("Invalid Parameters\n");
      }
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
  // std::string map_x_path = map_path_ + "/map_X.yaml";
  // std::string map_y_path = map_path_ + "/map_Y.yaml";
  cv::FileStorage fs(MAP_X_PATH, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    printf("load map_X failed, path: %s \n", MAP_X_PATH);
  }
  fs["map_X"] >> map_x;
  fs.release();
  fs.open(MAP_Y_PATH, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    printf("load map_Y failed, path: %s \n", MAP_Y_PATH);
  }

  fs["map_Y"] >> map_y;
  fs.release();
  gpumap_x_.upload(map_x);
  gpumap_y_.upload(map_y);
  initialized = true;

  pFuncs->fPreProcess = nullptr;
  pFuncs->fGPUProcess = gpu_process;
  pFuncs->fPostProcess = nullptr;
}