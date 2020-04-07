#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 16
#define N 5

__global__ void transpose_2D2D(float *A, float *B) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < N) {
     B[row + col * N] = A[col + row * N];
  }
}

void print_matrix(float *A) {
  for (int i = 0; i < N * N; i++) {
    printf("%.3f ", A[i]);
    if (i % N == N - 1) printf("\n");
  }
}

int check_transpose(float *A, float *B) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (B[j * N + i] != A[i * N + j]) return 0;
    }
  }
  
  return 1;
}

double diff_ms(struct timeval t0, struct timeval t1) {
  return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int main() {
  struct timeval t0, t1;
  int success;
  size_t size = sizeof(float) * N * N;

  // step 1: allocate memory in GPU
  float *d_A, *d_B;
  cudaMalloc((void**) &d_A, size);
  cudaMalloc((void**) &d_B, size);

  // step 2: allocate memory in CPU
  float *h_A, *h_B;
  h_A = (float *) malloc(size);
  h_B = (float *) malloc(size);

  for (int i = 0; i < N * N; i++) h_A[i] = (float) rand() / RAND_MAX;

  // step 3: transfer data from CPU to GPU
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  // steps 4-5: invoke kernel routine, transfer results
  dim3 block_structure (BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_structure (ceil(N * N / block_structure.x), ceil(N * N / block_structure.y));

  gettimeofday(&t0, NULL);
  transpose_2D2D<<<grid_structure, block_structure>>>(d_A, d_B);
  cudaDeviceSynchronize();  
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
  gettimeofday(&t1, NULL);

  success = check_transpose(h_A, h_B);

  printf(
    "2D2D -- status: %s, time: %lf ms\n", 
    success ? "success" : "failure", 
    diff_ms(t0, t1)
  );

  // step 6: free memory in GPU
  cudaFree(d_A);
  cudaFree(d_B);

  // step 7: free memory in CPU
  free(h_A);
  free(h_B);

  return 0;
}