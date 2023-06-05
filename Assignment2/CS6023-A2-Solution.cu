#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#define TILE_DIM 32
using namespace std;


// kernel for transpose
__global__ void transpose(int *d_matrixOut, int *d_matrixIn, int r, int c) {
    __shared__ int tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if(x<c && y<r)
        tile[threadIdx.y][threadIdx.x] = d_matrixIn[y*c + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if(x<r && y<c)
        d_matrixOut[y*r + x] = tile[threadIdx.x][threadIdx.y];
}

// kernel for matrix addition, A = A+B
__global__ void add_matrices(int *d_matrixA, int *d_matrixB) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    d_matrixA[index] += d_matrixB[index];
}

// kernel for matrix multiplication, C = AB
__global__ void multiply_matrices(int *d_matrixC, int *d_matrixA, int *d_matrixB, 
                                  int p, int q, int r) {
    __shared__ int A[TILE_DIM][TILE_DIM];
    __shared__ int B[TILE_DIM][TILE_DIM];

    int cValue = 0;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for(int k=0; k < (TILE_DIM+q-1)/TILE_DIM; k++) {
         if(k*TILE_DIM + threadIdx.x < q && y < p)
             A[threadIdx.y][threadIdx.x] = d_matrixA[y*q + k*TILE_DIM + threadIdx.x];
         else
             A[threadIdx.y][threadIdx.x] = 0;

         if(k*TILE_DIM + threadIdx.y < q && x < r)
             B[threadIdx.y][threadIdx.x] = d_matrixB[(k*TILE_DIM + threadIdx.y)*r + x];
         else
             B[threadIdx.y][threadIdx.x] = 0;

         __syncthreads();

         for(int n=0; n<TILE_DIM; ++n)
             cValue += A[threadIdx.y][n] * B[n][threadIdx.x];

         __syncthreads();
    }

    if(y<p && x<r)
        d_matrixC[y*r + x] = cValue;
}


// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
             int *h_matrixC, int *h_matrixD, int *h_matrixX) {
    // variable declarations
    int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
    int *d_matrixTemp;

    // allocate memory
    cudaMalloc(&d_matrixA, p * q * sizeof(int));
    cudaMalloc(&d_matrixB, q * p * sizeof(int));
    cudaMalloc(&d_matrixC, q * r * sizeof(int));
    cudaMalloc(&d_matrixD, s * r * sizeof(int));
    cudaMalloc(&d_matrixX, p * s * sizeof(int));

    // copy the values
    cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);

    // call the kernels to compute the transpose of B and D
    cudaMalloc(&d_matrixTemp, p * q * sizeof(int));
    transpose<<<dim3((p+31)/32, (q+31)/32), dim3(32, 32)>>>(d_matrixTemp, d_matrixB, q, p);
    cudaFree(d_matrixB); d_matrixB = d_matrixTemp;
    
    cudaMalloc(&d_matrixTemp, r * s * sizeof(int));
    transpose<<<dim3((r+31)/32, (s+31)/32), dim3(32, 32)>>>(d_matrixTemp, d_matrixD, s, r);
    cudaFree(d_matrixD); d_matrixD = d_matrixTemp;

    // call the kernel for matrix addition
    add_matrices<<<p, q>>>(d_matrixA, d_matrixB);

    // multiply matrices based on minimum number of operations required
    if((p*q*r + p*r*s) < (p*q*s + q*r*s)) {
        cudaMalloc(&d_matrixTemp, p * r * sizeof(int));
        multiply_matrices<<<dim3((r+31)/32, (p+31)/32), dim3(32, 32)>>>(d_matrixTemp, d_matrixA, d_matrixC, p, q, r);
        multiply_matrices<<<dim3((s+31)/32, (p+31)/32), dim3(32, 32)>>>(d_matrixX, d_matrixTemp, d_matrixD, p, r, s);
        cudaFree(d_matrixTemp);
    }
    else {
        cudaMalloc(&d_matrixTemp, q * s * sizeof(int));
        multiply_matrices<<<dim3((s+31)/32, (q+31)/32), dim3(32, 32)>>>(d_matrixTemp, d_matrixC, d_matrixD, q, r, s);
        multiply_matrices<<<dim3((s+31)/32, (p+31)/32), dim3(32, 32)>>>(d_matrixX, d_matrixA, d_matrixTemp, p, q, s);
        cudaFree(d_matrixTemp);
    }

    // copy the result back
    cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);

    // deallocate the memory
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    cudaFree(d_matrixD);
    cudaFree(d_matrixX);
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
    int p, q, r, s;
    int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
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
    fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

    // allocate memory and read input matrices
    matrixA = (int*) malloc(p * q * sizeof(int));
    matrixB = (int*) malloc(q * p * sizeof(int));
    matrixC = (int*) malloc(q * r * sizeof(int));
    matrixD = (int*) malloc(s * r * sizeof(int));
    readMatrix(inputFilePtr, matrixA, p, q);
    readMatrix(inputFilePtr, matrixB, q, p);
    readMatrix(inputFilePtr, matrixC, q, r);
    readMatrix(inputFilePtr, matrixD, s, r);

    // allocate memory for output matrix
    matrixX = (int*) malloc(p * s * sizeof(int));

    // call compute function to get the output matrix. it is expected that 
    // the compute function will store the result in matrixX.
    gettimeofday(&t1, NULL);
    compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    // print the time taken by the compute function
    seconds = t2.tv_sec - t1.tv_sec;
    microSeconds = t2.tv_usec - t1.tv_usec;
    printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

    // store the result into the output file
    outputFilePtr = fopen(outputFileName, "w");
    writeMatrix(outputFilePtr, matrixX, p, s);

    // close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

    // deallocate memory
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixD);
    free(matrixX);

    return 0;
}