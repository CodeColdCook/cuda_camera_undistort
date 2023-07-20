#include "cudaColorspace.h"
#include "cudaMappedMemory.h"
#include "imageIO.h"

#define TestConvertColor 0
#define TestSwamChanels 1

int main() {
  uchar3* imgInput = NULL;   // input is rgb8 (uchar3)
  uchar3* imgOutput = NULL;  // output is rgba32f (float4)

  int width = 0;
  int height = 0;

  // load the image as rgb8 (uchar3)
  if (!loadImage("/home/nvidia/pmtd_ws/deeprobotics-video-inference-streaming/"
                 "cuda_test/data/cata_test.jpg",
                 &imgInput, &width, &height))
    return -1;

  // allocate the output as rgba32f (float4), with the same width/height
  if (!cudaAllocMapped(&imgOutput, width, height)) return -1;

  // convert from rgb8 to rgba32f
  if (CUDA_FAILED(cudaConvertColor(imgInput, IMAGE_RGB8, imgOutput,
                                   IMAGE_BGR8, width, height)))
    return -1;  // an error or unsupported conversion occurred
  // if (CUDA_FAILED(cudaConvertColor(imgInput, IMAGE_RGB8, imgOutput,
  //                                  IMAGE_RGBA32F, width, height)))
  //   return -1;  // an error or unsupported conversion occurred

  saveImage(
      "/home/nvidia/pmtd_ws/deeprobotics-video-inference-streaming/"
      "cuda_test/data/cata_rgba.jpg",
      imgOutput, width, height);
}
