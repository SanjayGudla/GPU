#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


__global__ void calctranspose(int *input,int *output,int row, int col){
      __shared__ int sharedmem[32][32];

	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;

	  if (x < col && y < row)
	  {
		  sharedmem[threadIdx.y][threadIdx.x] = input[y * col + x];
	  }

	  __syncthreads();

	  x = blockIdx.y * blockDim.x + threadIdx.x;
	  y = blockIdx.x * blockDim.y + threadIdx.y;

	  if (x < row && y < col)
	  {
		  output[y * row + x] = sharedmem[threadIdx.x][threadIdx.y];
	  }
}
__global__ void colaescedmultkernel(int *input1, int *input2, int *input3, int *input4, int *output, int p, int q, int r)
{
	  __shared__ int shared1[32][32];
	  __shared__ int shared2[32][32];
	  __shared__ int shared3[32][32];
	  __shared__ int shared4[32][32];
	 

	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int l = ceil(q / float(32));
        int sum =0;
	  
	  for (int idx = 0; idx < l; idx++)
	  {

		  shared1[threadIdx.y][threadIdx.x] = 0;
		  shared3[threadIdx.y][threadIdx.x] = 0;
		  shared2[threadIdx.y][threadIdx.x] = 0;
		  shared4[threadIdx.y][threadIdx.x] = 0;

		  int temp1 = 32 * idx + threadIdx.x;
		  int temp2 = 32 * idx + threadIdx.y;

		  if (temp1 < q && y < p)
		  {
			  shared1[threadIdx.y][threadIdx.x] = input1[y * q + temp1];
			  shared3[threadIdx.y][threadIdx.x] = input3[y * q + temp1];
		  }

		  if (temp2 < q && x < r)
		  {
			  shared2[threadIdx.y][threadIdx.x] = input2[temp2 * r + x];
			  shared4[threadIdx.y][threadIdx.x] = input4[temp2 * r + x];
		  }
		  __syncthreads();

		  for (int n = 0; n < 32; n++)
		  {
			  sum += (shared1[threadIdx.y][n] * shared2[n][threadIdx.x]) + (shared3[threadIdx.y][n] * shared4[n][threadIdx.x]);
		  }
		  __syncthreads();
	  }
	  if(y< p && x<r){
		  output[y * r + x] = sum;
	  }
	  
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE, *d_matrixDT;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);
	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	int gridDimx,gridDimy;
    
	//kerenl 1
	cudaMalloc(&d_matrixDT, q * r * sizeof(int));
      gridDimx = ceil(q/(float)32);
	gridDimy = ceil(r/(float)32);
	dim3 grid1(gridDimx,gridDimy);
	calctranspose<<<grid1,dim3(32,32)>>>(d_matrixD,d_matrixDT,r,q);
	cudaFree(d_matrixD);
	d_matrixD = d_matrixDT;

	// kernel 2
	gridDimx = ceil(r / (float)32);
	gridDimy = ceil(p / (float)32);
	dim3 grid2(gridDimx, gridDimy);
	colaescedmultkernel<<<grid2, dim3(32, 32)>>>(d_matrixA, d_matrixB, d_matrixC,d_matrixD,d_matrixE, p, q, r);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}

// __global__ void multkernel(int *A, int *B, int *C, int *D, int *E, int p, int q, int r)
// {
// 	//   int i = blockIdx.x;
// 	//   int j = threadIdx.x;
// 	//   for (int k = 0; k < q; k++)
// 	//   {
// 	// 	  E[i * r + j] += (A[i * q + k] * B[k * r + j] + C[i * q + k] * D[j * q + k]);
// 	//   }

// 	for (int i = 0; i < p; i++)
// 	{

// 		for (int k = 0; k < q; k++)
// 		{
// 			for (int j = 0; j < r; j++)
// 			{
// 				E[i * r + j] += (A[i * q + k] * B[k * r + j] + C[i * q + k] * D[j * q + k]);
// 			}
// 		}
// 	}
// }
