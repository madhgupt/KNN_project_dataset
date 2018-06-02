/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

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

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const int *A, const int *B, int *C, int numElements_x , int numElements_y){
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int j = blockDim.y * blockIdx.y + threadIdx.y;

      if (i < numElements_x && j < numElements_y ){
            C[i*numElements_y + j] = A[i*numElements_y + j] + B[i*numElements_y + j];
      }
}


int main(void){

      cudaError_t err = cudaSuccess;

      int numElements_x = 10000;
      int numElements_y = 10000;
      int size = numElements_x * numElements_y * sizeof(int);
      printf("%d\n", size);
      printf("[Matrix addition of %d * %d matrix]\n", numElements_x , numElements_y);

      int *h_A = (int *)malloc(size);
      int *h_B = (int *)malloc(size);

      int *h_C = (int *)malloc(size);



      for(int i=0; i < numElements_x *numElements_y ; i++){
            h_A[i] = 1;
            h_B[i] = 1;
      }

      printf("Mat A:\n");
      for(int i=0; i < numElements_x ; i++){
            for (int j = 0; j < numElements_y; j++){
                  printf("%d\t", h_A[i*numElements_y + j]);
            }
            printf("\n");
      }

      printf("\n");

      printf("Mat B:\n");
      for(int i=0; i < numElements_x ; i++){
            for (int j = 0; j < numElements_y; j++){
                  printf("%d\t", h_B[i*numElements_y + j]);
            }
            printf("\n");
      }

      printf("here\n");

      int *d_A = NULL;
      cudaMalloc((void **)&d_A, size);

      int *d_B = NULL;
      cudaMalloc((void **)&d_B, size);

      int *d_C = NULL;
      cudaMalloc((void **)&d_C, size);


      printf("Copy input data from the host memory to the CUDA device\n");
      cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


      // int threadsPerBlock = 256;
      // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
      // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

      int threadsPerBlock = 32;
      dim3 dimBlock(threadsPerBlock , threadsPerBlock , 1);
      dim3 dimGrid(numElements_x + threadsPerBlock -1 , numElements_y + threadsPerBlock -1 , 1);

      vectorAdd<<<dimGrid, dimBlock >>>(d_A, d_B, d_C, numElements_x , numElements_y);

      err = cudaGetLastError();
      if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
      }

      printf("Copy output data from the CUDA device to the host memory\n");
      cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

      printf("Test PASSED\n");

      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);


      printf("Mat C:\n" );
      for(int i=0; i < numElements_x ; i++){
            for (int j = 0; j < numElements_y; j++){
                  printf("%d\t", h_C[i*numElements_y + j]);
            }
            printf("\n");
      }

      free(h_A);
      free(h_B);
      free(h_C);

      printf("Done\n");
      return 0;
}
