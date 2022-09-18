#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  const int TILE_WIDTH = 4;
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  int xo = blockIdx.x * TILE_WIDTH + tx;
  int yo = blockIdx.y * TILE_WIDTH + ty;
  int zo = blockIdx.z * TILE_WIDTH + tz;
  int xi = xo - (MASK_WIDTH / 2); int yi = yo - (MASK_WIDTH / 2); int zi = zo - (MASK_WIDTH / 2);
  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  float Pvalue = 0.0f;
  if ((xi >= 0) && (xi < x_size) && (yi >= 0) && (yi < y_size) && (zi >= 0) && (zi < z_size))
    tile[tz][ty][tx] = input[zi * x_size * y_size + yi * x_size + xi];
  else
    tile[tz][ty][tx] = 0.0f;
  __syncthreads();

  if ((tz < TILE_WIDTH) && (ty < TILE_WIDTH) && (tx < TILE_WIDTH)) {
    for (int i=0; i < MASK_WIDTH; i++) {
      for (int j=0; j < MASK_WIDTH; j++) {
        for (int k=0; k < MASK_WIDTH; k++) {
          Pvalue += Mc[i][j][k] * tile[i+tz][j+ty][k+tx];
        }
      }
    }

    if ((xo < x_size) && (yo < y_size) && (zo < z_size)) {
      output[zo * x_size * y_size + yo * x_size + xo + 3] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  // 1. Copy input (= N) to device memory
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  // 2. Copy kernel (= mask = M) to constant memory
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  // kernelLength => MASK_WIDTH ^ 3
  // Note: cannot be too large => A SM (streaming multiprocessors) can take up to 2048 threads.
  //       If BLOCK_WIDTH = 16, 16^3 = 4096 threads => cannot schedule to any SM.
  int BLOCK_WIDTH = 4; 
  dim3 dimGrid(ceil((1.0 * x_size)/BLOCK_WIDTH), ceil((1.0 * y_size)/BLOCK_WIDTH), ceil((1.0 * z_size)/BLOCK_WIDTH));
  dim3 dimBlock(BLOCK_WIDTH + MASK_WIDTH - 1, BLOCK_WIDTH + MASK_WIDTH - 1, BLOCK_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
