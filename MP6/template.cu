// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void FloatToUChar(float *inputImage, unsigned char *ucharImage, int width, int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int ii = blockIdx.z * (width * height) + row * width + col;
    ucharImage[ii] = (unsigned char) (255 * inputImage[ii]);
  }
}

__global__ void RGBtoGrayScale(unsigned char *rgbImage, unsigned char *grayImage, int width, int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int idx = row * width + col;
    unsigned char r = rgbImage[channels * idx];
    unsigned char g = rgbImage[channels * idx + 1];
    unsigned char b = rgbImage[channels * idx + 2];
    grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void ComputeHistogram(unsigned char *grayImage, unsigned int *histo, int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  if (idx < HISTOGRAM_LENGTH) {
    histo[idx] = 0;
  }  
  __syncthreads();

  while (idx < width * height) {
    atomicAdd( &(histo[grayImage[idx]]), 1);
    idx += stride;
  }
}

__global__ void HistoToCDF(unsigned int *histo, float *deviceCDF, int width, int height){
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < HISTOGRAM_LENGTH) {
    cdf[threadIdx.x] = histo[idx];
  }
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) {
      cdf[threadIdx.x] += cdf[threadIdx.x - stride];
    }
  }
  __syncthreads();

  if (idx < HISTOGRAM_LENGTH){
    deviceCDF[idx] = cdf[idx] / ((float)(width*height));
  }
}

__global__ void Equalization(unsigned char *deviceUCharImage, float *deviceCDF, int width, int height){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < height && col < width) {
    int ii = blockIdx.z * (width * height) + row * width + col;
    float cdfmin = deviceCDF[0];
    deviceUCharImage[ii] = (unsigned char) min(max(255*(deviceCDF[deviceUCharImage[ii]] - cdfmin)/(1.0 - cdfmin), 0.0), 255.0);
  }
}

__global__ void UCharToFloat(unsigned char *ucharImage, float *outputImage, int width, int height, int channels) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < height && col < width) {
    int ii = blockIdx.z * (width * height) + row * width + col;
    outputImage[ii] = (float) (ucharImage[ii]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceUCharImage; // RGB
  unsigned char *deviceGrayImage;  // Gray scale
  unsigned int  *deviceHistogram;  // Histogram
  float         *deviceCDF;        // CDF
  float         *deviceOutput;     // Output

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **) &deviceUCharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceCDF, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float));

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  int BLOCK_WIDTH = 16;
  dim3 dimGridToUChar(ceil((1.0 * imageWidth)/BLOCK_WIDTH), ceil((1.0 * imageHeight)/BLOCK_WIDTH), imageChannels);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  FloatToUChar<<<dimGridToUChar, dimBlock>>>(deviceInputImageData, deviceUCharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  dim3 dimGridToGray(ceil((1.0 * imageWidth)/BLOCK_WIDTH), ceil((1.0 * imageHeight)/BLOCK_WIDTH), 1);
  RGBtoGrayScale<<<dimGridToGray, dimBlock>>>(deviceUCharImage, deviceGrayImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dim3 dimGridHistogram(1, 1, 1);
  dim3 dimBlockHistogram(HISTOGRAM_LENGTH, 1, 1);
  ComputeHistogram<<<dimGridHistogram, dimBlockHistogram>>>(deviceGrayImage, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  HistoToCDF<<<dimGridHistogram, dimBlockHistogram>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  Equalization<<<dimGridToUChar, dimBlock>>>(deviceUCharImage, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  UCharToFloat<<<dimGridToUChar, dimBlock>>>(deviceUCharImage, deviceOutput, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceUCharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  cudaFree(deviceOutput);

  return 0;
}
