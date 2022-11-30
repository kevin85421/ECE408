#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MASK_WIDTH 7

__constant__ float mask_constant[3136];

__global__ void conv_forward_kernel_tile(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_constant[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x; int ty = threadIdx.y;

    int W_size = ceil((1.0 * Width_out)/TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + ty;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + tx;
    int b = blockIdx.z;

    __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    float acc = 0.0f;
    for (int c=0; c < Channel; c++) {
        if ((w >= 0) && (w < Width) && (h >= 0) && (h < Height)) {
            input_tile[ty][tx] = in_4d(b, c, h, w);
        } else {
            input_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        if ((ty < TILE_WIDTH) && (tx < TILE_WIDTH)) {
            for (int p=0; p < K; p++) {
                for (int q=0; q < K; q++)
                    acc += input_tile[ty+p][tx+q] * mask_4d(m, c, p, q); 
            }
        }
        __syncthreads();
    }

    if ((ty < TILE_WIDTH) && (tx < TILE_WIDTH) && (h < Height_out) && (w < Width_out)) {
      out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_constant[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_size = ceil((1.0 * Width_out)/TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;

    float acc = 0.0f;
    if (h < Height_out && w < Width_out) {
        for (int c=0; c < Channel; c++) {
            for (int p=0; p < K; p++) {
                for (int q=0; q < K; q++)
                    acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
            }
        }
        out_4d(b, m, h, w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void matrixMultiply(const float *A, const float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int Channel, int Height, int Width, int K) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH]; 

  int batch = blockIdx.z;
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  #define in_4d(i3, i2, i1, i0) B[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
  for (int q=0; q < (ceil((float) numAColumns/ TILE_WIDTH)); ++q) {
    if (row < numARows && (q*TILE_WIDTH + tx) < numAColumns)
      subTileA[ty][tx] = A[row * numAColumns + q * TILE_WIDTH + tx];
    else
      subTileA[ty][tx] = 0;

    int input_channel = (q*TILE_WIDTH + ty) / (K * K);
    int input_height = col / (Width - K + 1);
    int input_width  = col % (Width - K + 1);
    int input_dh = ((q*TILE_WIDTH + ty) % (K * K)) / K;
    int input_dw = ((q*TILE_WIDTH + ty) % (K * K)) % K;

    if (col < numBColumns && (q*TILE_WIDTH + ty) < numBRows)
      subTileB[ty][tx] = in_4d(batch, input_channel, input_height + input_dh, input_width + input_dw);
    else
      subTileB[ty][tx] = 0;
    __syncthreads();

    for (int k=0; k < TILE_WIDTH; ++k)
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();
  }
  #undef in_4d
  if (row < numCRows && col < numCColumns)
    C[batch * numCRows * numCColumns + row * numCColumns + col] = Pvalue;
}


void unroll_seq(int B, int C, int H, int W, int K, const float* X, float* X_unroll)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int H_unroll = C * K * K;

    #define in_4d(i3, i2, i1, i0) X[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    for (int b=0; b < B; ++b) {
        for (int c=0; c < C; ++c) {
            int w_base = c * (K*K);
            for (int p=0; p < K; ++p) {
                for (int q=0; q < K; ++q) {
                    for (int h=0; h < H_out; ++h) {
                        for (int w=0; w < W_out; ++w) {
                            int h_unroll = w_base + p * K + q;
                            int w_unroll = h * W_out + w;
                            X_unroll[b * W_unroll * H_unroll + h_unroll * W_unroll + w_unroll] = in_4d(b, c, h+p, w+q);
                        }
                    }
                }
            }
        }
    }
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    int input_size = Batch * Channel * Height * Width;
    int mask_size = Map_out * Channel * K * K;

    cudaMalloc((void **) device_output_ptr, output_size * sizeof(float));
    cudaMalloc((void **) device_input_ptr, input_size * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void unroll_kernel(int Channel, int Height, int Width, int K, const float *device_input, float *X_unroll)
{
    int c, s, h_out, w_out, h_unroll, w_unroll, h_base, p, q;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    int W_unroll = H_out * W_out;

    #define in_3d(i2, i1, i0) device_input[(i2) * (Height * Width) + (i1) * (Width) + i0]
    if (t < Channel * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = h_out * W_out + w_out;
        h_base = c * K * K;
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                h_unroll = h_base + p * K + q;
                X_unroll[h_unroll * W_unroll + w_unroll] = in_3d(c, h_out + p, w_out + q); 
            }
        }
    }
    #undef in_3d
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // printf("Batch: %d, Map_out: %d, Channel: %d, Height: %d, Width: %d, K: %d\n", Batch, Map_out, Channel, Height, Width, K);
    int W_out = Width - K + 1;
    int H_out = Height - K + 1;

    // conv filter: m * (ck^2); X_unroll: (ck^2) * (H_out * W_out) => output feature: m * (H_out * W_out)
    dim3 dimGrid(ceil((1.0 * H_out * W_out)/TILE_WIDTH), ceil((1.0 * Map_out)/TILE_WIDTH), Batch);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //@@ Launch the GPU Kernel here
    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns; 

    matrixMultiply<<<dimGrid, dimBlock>>>(device_mask, device_input, device_output,
        numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1); 
    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask); 
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
