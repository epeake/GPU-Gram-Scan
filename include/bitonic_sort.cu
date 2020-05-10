#include <iostream>
#include <vector>
#include <algorithm>

#include "cuda-util.h"
#include "gpu_graham_scan.h"

const uint kMaxThreads = 1024;
const uint kChunksPerBlock = 32;
const uint kTotalSMs = 46;
const uint kMaxThreadsSpan = kMaxThreads * kTotalSMs * 2;

/*
 * see if p1 is "less" than p2
 */
template <class Num_Type>
__device__ bool comparePoints(const gpu_graham_scan::Point<Num_Type> p1,
                              const gpu_graham_scan::Point<Num_Type> p2) {
  // cross product of 2 points
  Num_Type x_product = (p1.x_ * p2.y_) - (p2.x_ * p1.y_);

  if (x_product > 0) {
    return true;
  }
  if (x_product == 0) {
    Num_Type sq_mag_p1 = (p1.x_ * p1.x_) + (p1.y_ * p1.y_);
    Num_Type sq_mag_p2 = (p2.x_ * p2.x_) + (p2.y_ * p2.y_);
    return sq_mag_p1 < sq_mag_p2;
  }
  return false;
}

template <class Num_Type>
  __global__ void BuildBitonic(int n, gpu_graham_scan::Point<Num_Type>* d_points, int blocks_per_chunk, int chunk_size){
    int chunk_id = blockIdx.x / blocks_per_chunk; // could also use second BlockIdx.y
    int chunk_start = chunk_id * chunk_size * 2; //assuming chunk size is all points to be compared in a chunk
    int chunk_end = chunk_start + chunk_size*2 -1;
    
    int block_start = chunk_start + (blockIdx.x % blocks_per_chunk) * chunk_size/blocks_per_chunk;
    int block_end = block_start + chunk_size/blocks_per_chunk;
    
    int elt_of_interest = block_start + threadIdx.x;
    
    if (threadIdx.x < chunk_size){
      printf("\nchunk end: %d\n", chunk_end);
      printf("\ntest comparator: %d\n", comparePoints(d_points[1], d_points[2]));
      int partner_elt = chunk_end - threadIdx.x;
      printf("blockIdx.x: %d\tthreadIdx.x: %d\ncomparing elements: %d, %d\n", blockIdx.x, threadIdx.x, elt_of_interest, partner_elt);

      gpu_graham_scan::Point<Num_Type> current = d_points[elt_of_interest];
      gpu_graham_scan::Point<Num_Type> partner = d_points[partner_elt];
      
      printf("\nannouncing swaps:\n");
      printf("current point %d: %f, %f\n", elt_of_interest, current.x_, current.y_);
      printf("partner point %d: %f, %f\n", partner_elt, partner.x_, partner.y_);
      printf("\ncomparePoints block %d, thread %d: %d == %d\n ", blockIdx.x, threadIdx.x, comparePoints(d_points[elt_of_interest], d_points[partner_elt]), comparePoints(current, partner));
      printf("%f, %f < %f, %f : %d\n", current.x_, current.y_, partner.x_, partner.y_, comparePoints(current, partner));
      if (!comparePoints(current, partner)) {
        printf("swapping index %d, %d\n", elt_of_interest, partner_elt);
        d_points[elt_of_interest] = partner;
        d_points[partner_elt] = current;
        
      } 
      // else {
      //   d_points[elt_of_interest] = current;
      //   d_points[partner_elt] = partner;
      // }
    }
    
  }

  template <class Num_Type>
  __global__ void  BitonicSortPointsKernel(int n, gpu_graham_scan::Point<Num_Type>* d_points, int blocks_per_chunk, int chunk_size){
    int chunk_id = blockIdx.x / blocks_per_chunk; // could also use second BlockIdx.y
    int chunk_start = chunk_id * chunk_size * 2; //assuming chunk size is all points to be compared in a chunk
    int chunk_end = chunk_start + chunk_size*2 -1;
    
    int block_start = chunk_start + (blockIdx.x % blocks_per_chunk) * chunk_size/blocks_per_chunk;
    int block_end = block_start + chunk_size/blocks_per_chunk;
    
    int elt_of_interest = block_start + threadIdx.x;
    int step_size = chunk_size;


    if (threadIdx.x < chunk_size){
      printf("\nchunk end: %d\n", chunk_end);
      int partner_elt = elt_of_interest + step_size;
      printf("blockIdx.x: %d\tthreadIdx.x: %d\ncomparing elements: %d, %d\n", blockIdx.x, threadIdx.x, elt_of_interest, partner_elt);

      gpu_graham_scan::Point<Num_Type> current = d_points[elt_of_interest];
      gpu_graham_scan::Point<Num_Type> partner = d_points[partner_elt];
      
      printf("\nannouncing swaps:\n");
      if (comparePoints(current, partner)) {
        d_points[elt_of_interest] = current;
        d_points[partner_elt] = partner;
      } else {
        printf("swapping index %d, %d\n", elt_of_interest, partner_elt);
        d_points[elt_of_interest] = partner;
        d_points[partner_elt] = current;
      }
    }

  }



template <class Num_Type>
void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<Num_Type>> points) {
  size_t n_points = points.size();

  // underlying array of points to put onto GPU
  gpu_graham_scan::Point<Num_Type>* points_arr = points.data();

  // Allocate device data
  gpu_graham_scan::Point<Num_Type>* d_points;

  cudaErrorCheck(cudaMalloc(
      &d_points, n_points * sizeof(gpu_graham_scan::Point<Num_Type>)));

  // points to device
  cudaErrorCheck(cudaMemcpy(d_points, points_arr,
                            n_points * sizeof(gpu_graham_scan::Point<Num_Type>),
                            cudaMemcpyHostToDevice));

  // round up to the the power of 2 to get our upper bound
  size_t upper_bound = n_points;
  uint power = 0;
  while (upper_bound) {
    upper_bound >>= 1;
    power++;
  }
  size_t curr_bound = 1 << (power - 1);
  upper_bound = (curr_bound < n_points) ? (curr_bound << 1) : curr_bound;

  double start_time = CycleTimer::currentSeconds();
  // switch when chunk count => threads hits 256
  for (size_t i = 2, j = i; i <= upper_bound; i *= 2, j = i) {
    size_t chunks = (n_points + j - 1) / j;
    size_t chunk_size = j >> 1;

    int blocks_per_chunk = (chunk_size / kMaxThreads) + 1;


    // each thread gets its own chunk
    // each block has multiple chunks
    std::cout << "\nblocks_per_chunk:" << blocks_per_chunk << "\n";
    std::cout << "chunks:" << chunks << "\n";
    std::cout << "chunk size: " << chunk_size << "\n";
    std::cout << "chunk_size/blocks_per_chunk: " << chunk_size/blocks_per_chunk << "\n\n";
    std::cout << "\n************ Build Bitonic called ************\n";
    BuildBitonic<<<dim3(blocks_per_chunk*chunks), dim3(kMaxThreads)>>>(n_points, d_points, blocks_per_chunk, chunk_size);
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(points_arr, d_points,
      n_points * sizeof(gpu_graham_scan::Point<Num_Type>),
      cudaMemcpyDeviceToHost));

    for(int i=0;i<n_points;i++){
      Point<Num_Type> cp = points_arr[i];
      std::cout << "point "<< i << ": " << cp.x_ << " , " << cp.y_ << "\n";
    }
    std::cout << "\n";
    

    // wait for build to finish
    cudaErrorCheck(cudaDeviceSynchronize());
    j >>= 1; // j is the size of a chunk in this step
    while (j > 1) {
      int blocks_per_chunk = (j / (2*kMaxThreads)) + 1;
      size_t chunks = (n_points + j - 1) / j;
      size_t chunk_size = j >> 1;
      // sort bionic
      std::cout << "j: " << j;
      std::cout << "\nblocks_per_chunk:" << blocks_per_chunk << "\n";
      std::cout << "chunks:" << chunks << "\n";
      std::cout << "chunk size: " << chunk_size << "\n";
      std::cout << "chunk_size/blocks_per_chunk: " << chunk_size/blocks_per_chunk << "\n\n";
      std::cout << "\n************ BitonicSortPointsKernel called ************\n";
      BitonicSortPointsKernel<<<dim3(blocks_per_chunk*chunks), dim3(kMaxThreads)>>>(n_points, d_points, blocks_per_chunk, chunk_size);
      cudaErrorCheck(cudaDeviceSynchronize());
      cudaErrorCheck(cudaMemcpy(points_arr, d_points,
        n_points * sizeof(gpu_graham_scan::Point<Num_Type>),
        cudaMemcpyDeviceToHost));

      for(int i=0;i<n_points;i++){
        Point<Num_Type> cp = points_arr[i];
        std::cout << "point "<< i << ": " << cp.x_ << " , " << cp.y_ << "\n";
      }
      std::cout << "\n";


      j >>= 1;
    }
  }
  double end_time = CycleTimer::currentSeconds();
  printf("[Build Bitonic]:\t%.3f ms\n", (end_time - start_time) * 1000);

  // Copy points back to host points to device
  cudaErrorCheck(cudaMemcpy(points_arr, d_points,
                            n_points * sizeof(gpu_graham_scan::Point<Num_Type>),
                            cudaMemcpyDeviceToHost));

  // Cleanup device data
  cudaErrorCheck(cudaFree(d_points));
}

template void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<int>> points);

template void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<double>> points);

template void __global__ BuildBitonic(int n, gpu_graham_scan::Point<int>* d_points, int blocks_per_chunk, int chunk_size);

template void __global__ BuildBitonic(int n, gpu_graham_scan::Point<double>* d_points, int blocks_per_chunk, int chunk_size);

template __device__ bool comparePoints(const gpu_graham_scan::Point<int> p1,
                                       const gpu_graham_scan::Point<int> p2);

template __device__ bool comparePoints(const gpu_graham_scan::Point<double> p1,
                                       const gpu_graham_scan::Point<double> p2);

template __global__ void  BitonicSortPointsKernel(int n, gpu_graham_scan::Point<int>* d_points, int blocks_per_chunk, int chunk_size);

template __global__ void  BitonicSortPointsKernel(int n, gpu_graham_scan::Point<double>* d_points, int blocks_per_chunk, int chunk_size);