#include <iostream>
#include <stdio.h>
#include <unordered_set>
#include <cuda.h>
#include<bits/stdc++.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

struct requests
{
  int *request_id;
  int *computer_no;
  int *facilty_room;
  int *start_slot;
  int *end_slot;
};
//*******************************************

// Write down the kernels here

__global__ void Booking(requests d_reqs, int R, int *d_distinct_computer_arr, int *d_succ_reqs, int *d_bookingarr, int *d_success)
{
  int compcenter_id = d_distinct_computer_arr[blockIdx.x];
  int faciltiy_id = threadIdx.x;

  for (int i = 0; i < R; i++)
  {
    if (d_reqs.computer_no[i] == compcenter_id && d_reqs.facilty_room[i] == faciltiy_id)
    {
      int st = d_reqs.start_slot[i];
      int end = d_reqs.end_slot[i];
      int idx = (d_reqs.computer_no[i] * 30  + threadIdx.x) * 25;

      int breakindx=-1;
      for (int j = idx+st; j < idx+end; j++)
      {
        if(d_bookingarr[j]<=0){
          breakindx=j;
          break;
        }
        d_bookingarr[j]--;
      }
      if(breakindx!=-1){
        for (int k = st + idx; k < breakindx; k++)
        {
          d_bookingarr[k]++;
        }
      }
      else
      {
        atomicAdd(&d_succ_reqs[compcenter_id], 1);
        atomicAdd(d_success, 1);
      }
    }
  }
}


//***********************************************

int main(int argc, char **argv)
{
  // variable declarations...
  int N, *centre, *facility, *capacity, *fac_ids, *succ_reqs, *tot_reqs;

  FILE *inputfilepointer;

  // File Opening for read
  char *inputfilename = argv[1];
  inputfilepointer = fopen(inputfilename, "r");

  if (inputfilepointer == NULL)
  {
    printf("input.txt file failed to open.");
    return 0;
  }

  fscanf(inputfilepointer, "%d", &N); // N is number of centres

  // Allocate memory on cpu
  centre = (int *)malloc(N * sizeof(int));           // Computer  centre numbers
  facility = (int *)malloc(N * sizeof(int));         // Number of facilities in each computer centre
  fac_ids = (int *)malloc(max_P * N * sizeof(int));  // Facility room numbers of each computer centre
  capacity = (int *)malloc(max_P * N * sizeof(int)); // stores capacities of each facility for every computer centre

  int success = 0;                            // total successful requests
  int fail = 0;                               // total failed requests
  tot_reqs = (int *)malloc(N * sizeof(int));  // total requests for each centre
  succ_reqs = (int *)malloc(N * sizeof(int)); // total successful requests for each centre

  // Input the computer centres data
  int k1 = 0, k2 = 0;
  for (int i = 0; i < N; i++)
  {
    fscanf(inputfilepointer, "%d", &centre[i]);
    fscanf(inputfilepointer, "%d", &facility[i]);

    for (int j = 0; j < facility[i]; j++)
    {
      fscanf(inputfilepointer, "%d", &fac_ids[k1]);
      k1++;
    }
    for (int j = 0; j < facility[i]; j++)
    {
      fscanf(inputfilepointer, "%d", &capacity[k2]);
      k2++;
    }
  }

  // variable declarations
  int *req_id, *req_cen, *req_fac, *req_start, *req_slots, *req_end; // Number of slots requested for every request

  // Allocate memory on CPU
  int R;
  fscanf(inputfilepointer, "%d", &R);           // Total requests
  req_id = (int *)malloc((R) * sizeof(int));    // Request ids
  req_cen = (int *)malloc((R) * sizeof(int));   // Requested computer centre
  req_fac = (int *)malloc((R) * sizeof(int));   // Requested facility
  req_start = (int *)malloc((R) * sizeof(int)); // Start slot of every request
  req_slots = (int *)malloc((R) * sizeof(int)); // Number of slots requested for every request
  req_end = (int *)malloc((R) * sizeof(int));
  // Input the user request data
  int totalreqs = 0;
  for (int j = 0; j < R; j++)
  {
    fscanf(inputfilepointer, "%d", &req_id[j]);
    fscanf(inputfilepointer, "%d", &req_cen[j]);
    fscanf(inputfilepointer, "%d", &req_fac[j]);
    fscanf(inputfilepointer, "%d", &req_start[j]);
    fscanf(inputfilepointer, "%d", &req_slots[j]);
    req_end[j] = req_start[j] + req_slots[j];
    tot_reqs[req_cen[j]] += 1;
    totalreqs += 1;
  }

  //*********************************

  // creating structure of arrays for requests
  requests d_reqs;
  cudaMalloc(&d_reqs.request_id, (R) * sizeof(int));
  cudaMalloc(&d_reqs.computer_no, (R) * sizeof(int));
  cudaMalloc(&d_reqs.facilty_room, (R) * sizeof(int));
  cudaMalloc(&d_reqs.start_slot, (R) * sizeof(int));
  cudaMalloc(&d_reqs.end_slot, (R) * sizeof(int));

  cudaMemcpy(d_reqs.request_id, req_id, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reqs.computer_no, req_cen, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reqs.facilty_room, req_fac, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reqs.start_slot, req_start, (R) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_reqs.end_slot, req_end, (R) * sizeof(int), cudaMemcpyHostToDevice);

  // creating booking array
  int *h_bookingarr = (int *)malloc((N * 30 * 25) * sizeof(int));
  int *d_bookingarr;

  int l = 0;
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < facility[i]; j++)
    {
      for (int k = 0; k <= 24; k++)
      {
        h_bookingarr[(i * 30  + j) * 25 + k] = capacity[l];
      }
      l++;
    }
  }
  cudaMalloc(&d_bookingarr, (N * 30 * 25) * sizeof(int));
  cudaMemcpy(d_bookingarr, h_bookingarr, (N * 30 * 25) * sizeof(int), cudaMemcpyHostToDevice);

  // finding launch param
  unordered_set<int> m(req_cen,req_cen+R);
  int sz = (int)m.size();
  int *distinct_computer_arr = (int *)malloc(sizeof(int) * sz);
  std::copy(m.begin(), m.end(), distinct_computer_arr);
  int *d_distinct_computer_arr;
  cudaMalloc(&d_distinct_computer_arr, sizeof(int) * sz);
  cudaMemcpy(d_distinct_computer_arr, distinct_computer_arr, sizeof(int) * sz, cudaMemcpyHostToDevice);

  // Call the kernels here
  int *d_succ_reqs;
  cudaMalloc(&d_succ_reqs, (N) * sizeof(int));
  int *d_success;
  cudaMalloc(&d_success, sizeof(int));
  cudaMemset(d_success, 0, sizeof(int));

  Booking<<<sz, 30>>>(d_reqs, R, d_distinct_computer_arr, d_succ_reqs, d_bookingarr, d_success);
  cudaDeviceSynchronize();
  cudaMemcpy(succ_reqs, d_succ_reqs, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&success, d_success, sizeof(int), cudaMemcpyDeviceToHost);
  fail = totalreqs - success;

  //********************************

  // Output
  char *outputfilename = argv[2];
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");

  fprintf(outputfilepointer, "%d %d\n", success, fail);
  for (int j = 0; j < N; j++)
  {
    fprintf(outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j] - succ_reqs[j]);
  }
  fclose(inputfilepointer);
  fclose(outputfilepointer);
  cudaDeviceSynchronize();
  return 0;
}