#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void atomicHistogram(int * Histogram, const int * data)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    int value = data[id];
    int Histovalue = DetectRange(value);
    atomicAdd(&Histogram[Histovalue], 1);
}

int main(void)
{
    int numElements = 1000; 
    int threadsPerBlock = 512; 
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    size_t size = numElements * sizeof(float);

    float *host_histogram = (float *)malloc(size);
    float *host_data = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        host_histogram[i] = rand()/(float)RAND_MAX;
        host_data[i] = rand()/(float)RAND_MAX;
    }

    float *device_histogram = NULL;  cudaMalloc((void **)&device_histogram, size);
    float *device_data = NULL;  cudaMalloc((void **)&device_data, size);

    cudaMemcpy(device_histogram, host_histogram, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start,stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    atomicHistogram<<<blocksPerGrid, threadsPerBlock>>>(device_histogram, device_data);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(host_histogram, device_histogram, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double memXFers=5*4*COLUMNS*ROWS;
    memXFers/=1024*1024*1024;

    printf("GPU: %f ms bandwidth %g GB/s",ms, memXFers/(ms/1000.0));
    printf("\n CPU : %g ms bandwidth %g GB/s",mtime, memXFers/(mtime/1000.0));

    cudaFree(device_histogram);
    cudaFree(device_data);

    free(host_histogram);
    free(host_data);

    return 0;
}