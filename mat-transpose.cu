#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 4
#define N 2000

#define CHUNK_SIZE (BLOCK_SIZE * BLOCK_SIZE)

__global__ void transpose_2D2D(float *A, float *B) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < N && row < N) {
     B[row + col * N] = A[col + row * N];
  } 
}

__global__ void transpose_shmem(float *A, float *B) {
  __shared__ float chunk[CHUNK_SIZE * CHUNK_SIZE];

  int col_chunk = blockIdx.x * CHUNK_SIZE;
  int row_chunk = blockIdx.y * CHUNK_SIZE;

  int col = col_chunk;
  int row_offset = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int chunk_offset = CHUNK_SIZE * row_offset;
  int row = row_chunk + row_offset;
  int A_offset = row * N + col;

  for (int k = 0; k < CHUNK_SIZE; k++) {
    chunk[chunk_offset + k] = A[A_offset + k];
  }

  __syncthreads();

  int row_out = row_chunk;
  int col_out = col_chunk + row_offset;
  int out = col_out * N + row_out;

  for (int k = 0; k < CHUNK_SIZE; k++) {
    B[out + k] = chunk[row_offset + CHUNK_SIZE * k];
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

void print_results(char* label, float *A, float *B, struct timeval t0, struct timeval t1) {
  int success = check_transpose(A, B);

  printf(
    "%s -- status: %s, time: %lf ms\n", 
    label,
    success ? "success" : "failure", 
    diff_ms(t0, t1)
  );
}

int main() {
  struct timeval t0, t1;
  size_t size = sizeof(float) * N * N;

  // step 1: allocate memory in GPU
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**) &d_A, size);
  cudaMalloc((void**) &d_B, size);
  cudaMalloc((void**) &d_C, size);

  // step 2: allocate memory in CPU
  float *h_A, *h_B;
  h_A = (float *) malloc(size);
  h_B = (float *) malloc(size);

  for (int i = 0; i < N * N; i++) h_A[i] = (float) rand() / RAND_MAX;

  // step 3: transfer data from CPU to GPU
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

  // steps 4-5: invoke kernel routine, transfer results

  // 2D2D implementation
  gettimeofday(&t0, NULL);

  dim3 block_2D2D (BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_2D2D (ceil((float) N / block_2D2D.x), ceil((float) N / block_2D2D.y));

  transpose_2D2D<<<grid_2D2D, block_2D2D>>>(d_A, d_B);
  cudaDeviceSynchronize();  
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

  gettimeofday(&t1, NULL);
  print_results("2D2D", h_A, h_B, t0, t1);

  // shared memory implementation
  gettimeofday(&t0, NULL);

  dim3 block_shmem (BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_shmem (ceil((float) N / CHUNK_SIZE), ceil((float) N / CHUNK_SIZE));

  transpose_shmem<<<grid_shmem, block_shmem>>>(d_A, d_C);
  cudaDeviceSynchronize();
  cudaMemcpy(h_B, d_C, size, cudaMemcpyDeviceToHost);

  gettimeofday(&t1, NULL);
  print_results("shared memory", h_A, h_B, t0, t1);

  // step 6: free memory in GPU
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // step 7: free memory in CPU
  free(h_A);
  free(h_B);

  return 0;
}