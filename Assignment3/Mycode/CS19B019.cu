/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
//1)    init kernel
__global__ void initKernel(int V, int *init_array, int init_value)
{ // intializes one 1D array with init val
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < V)
    {
        init_array[id] = init_value;
    }
}

//2)    Finding level Zero vertices
__global__ void findLevelZero(int V,int *d_apr, int* d_state,int *d_activeVertex,int *end_new){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < V && d_apr[id]==0)
    {
        d_state[id]=1;
        atomicAdd(&d_activeVertex[0], 1);
        atomicMax(end_new, id);
    }
}

//3)   Level order traversal of the graph


__global__ void bfs(int V, int level, int *d_offset, int *d_csrList,int *d_aid,int *d_state,int start_curr,int end_curr,int *end_new)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid = tid + start_curr;
    if (tid > end_curr)
    {
        return;
    }
    int end1 = -1;
    for (int edge = d_offset[tid]; edge < d_offset[tid + 1]; edge++)
    {
        int nbr = d_csrList[edge];
        end1=max(end1,nbr);
        if(d_state[tid]==1){
            atomicAdd(&d_aid[nbr],1);
        }
    }
    atomicMax(end_new,end1);   
}

// 3)  activation kernel
__global__ void simulate_activation(int V, int level, int *d_offset, int *d_csrList, int *d_aid, int *d_state, int *d_apr, int *d_activeVertex, int start_curr, int end_curr)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid = tid + start_curr;
    if (tid > end_curr)
    {
        return;
    }
    if(d_aid[tid]>=d_apr[tid] && d_state[tid]==0){
        d_state[tid]=1;
        atomicAdd(&d_activeVertex[level],1);
    }
}

//4) deactivation kernel
__global__ void simulate_deactivation(int V, int level, int *d_offset, int *d_csrList, int *d_aid, int *d_state, int *d_activeVertex, int start_curr, int end_curr)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    tid = tid + start_curr;
    if (tid > end_curr)
    {
        return;
    }
    if(d_state[tid]==1){

        if ((tid - 1 >= start_curr && d_state[tid - 1] == 0) && (tid + 1 <= end_curr && d_state[tid + 1] == 0))
        {
            d_state[tid] = 0;
            atomicAdd(&d_activeVertex[level], -1);
        }
       
    }
}
/**************************************END*************************************************/

//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc, char **argv)
{
    // Variable declarations
    int V; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    // Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    // Parsing the graph to create csr list
    g.parseGraph();

    // Reading graph info
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();

    // Variable for CSR format on host
    int *h_offset;  // for csr offset
    int *h_csrList; // for csr
    int *h_apr;     // active point requirement

    // reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();
    h_apr = g.get_aprArray();

    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; // activation point requirement array
    int *d_aid; // acive in-degree array
    // Allocating memory on device
    cudaMalloc(&d_offset, (V + 1) * sizeof(int));
    cudaMalloc(&d_csrList, E * sizeof(int));
    cudaMalloc(&d_apr, V * sizeof(int));
    cudaMalloc(&d_aid, V * sizeof(int));

    // copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V * sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int *)malloc(L * sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L * sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
    cudaMalloc(&d_activeVertex, L*sizeof(int));

   
    /***Important***/

    // Initialize d_aid array to zero for each vertex
    //  Make sure to use comments
    // 1.   Initialize d_aid array to zero for each vertex
    const int threadsPerBlock = 512;
    int numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
    initKernel<<<numBlocks, threadsPerBlock>>>(V, d_aid, 0);

    /***END***/
    double starttime = rtclock();

    /*********************************CODE AREA*****************************************/
    

    // array to store state of vertex that is whether a vertex is active or not
    int *d_state;
    cudaMalloc(&d_state, V * sizeof(int));

    // 2.    variable for storing state of each vertex
    initKernel<<<numBlocks, threadsPerBlock>>>(V, d_state, 0);

    // 3.    traverse to find level 0 vertices
    int *end_new =(int*)malloc(sizeof(int));
    end_new[0] = -1;
    int *d_end_new;
    cudaMalloc(&d_end_new,sizeof(int));
    cudaMemcpy(d_end_new, end_new, sizeof(int), cudaMemcpyHostToDevice);
    findLevelZero<<<numBlocks, threadsPerBlock>>>(V, d_apr, d_state, d_activeVertex,d_end_new);
    cudaMemcpy(end_new, d_end_new, sizeof(int), cudaMemcpyDeviceToHost);

    //4. BFS to find level of each node
    int start_curr = 0;
    int end_curr = end_new[0];

    for(int level = 1;level<L;level++){
        end_new[0] =-1;
        int blocks = ceil((end_curr-start_curr+1)/1024.0);
        cudaMemcpy(d_end_new, end_new, sizeof(int), cudaMemcpyHostToDevice);
        bfs<<<blocks,1024>>>(V, level, d_offset, d_csrList, d_aid, d_state, start_curr, end_curr,d_end_new);
        cudaMemcpy(end_new, d_end_new, sizeof(int), cudaMemcpyDeviceToHost);
        start_curr = end_curr + 1;
        end_curr = end_new[0];
        blocks = ceil((end_curr - start_curr + 1) / 4.0);
        simulate_activation<<<blocks, 1024>>>(V, level, d_offset, d_csrList, d_aid, d_state, d_apr, d_activeVertex, start_curr, end_curr);
        simulate_deactivation<<<blocks, 1024>>>(V, level, d_offset, d_csrList, d_aid, d_state, d_activeVertex, start_curr, end_curr);
    }

    /********************************END OF CODE AREA**********************************/
    double endtime = rtclock();
    printtime("GPU Kernel time: ", starttime, endtime);

    // --> Copy C from Device to Host
    cudaMemcpy(h_activeVertex, d_activeVertex, L * sizeof(int), cudaMemcpyDeviceToHost);
    char outFIle[30] = "./output.txt";
    printResult(h_activeVertex, L, outFIle);
    if (argc > 2)
    {
        for (int i = 0; i < L; i++)
        {
            printf("level = %d , active nodes = %d\n", i, h_activeVertex[i]);
        }
    }

    return 0;
}