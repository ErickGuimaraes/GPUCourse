
/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
__global__ void mykernel(void)
{
 printf("hello word from GPU \n"); 
}

/**
 * Host main routine
 */
int
main(void)
{

    	mykernel<<< 1,10 >>>();
	cudaDeviceSynchronize();
    	printf("hello word \n");
    	return 0;
}

