#include <iostream>
#include <vector>

#include "cuda-util.h"
#include "gpu_graham_scan.h"

const uint kMaxThreads = 1024;
const uint kChunksPerBlock = 256;
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
    int chunk_start = chunk_id * chunk_size; //assuming chunk size is all points to be compared in a chunk
    // int chunk_end = chunk_start + chunk_size;
    
    int block_start = chunk_start + (blockIdx.x % blocks_per_chunk) * chunk_size/blocks_per_chunk;
    int block_end = block_start + chunk_size/blocks_per_chunk;
    
    int elt_of_interest = block_start + threadIdx.x;
    
    if (elt_of_interest < block_start + chunk_size / 2) {
      int partner_elt = block_end - threadIdx.x;
      gpu_graham_scan::Point<Num_Type> current = d_points[elt_of_interest];
      gpu_graham_scan::Point<Num_Type> partner = d_points[partner_elt];
      
      // gpu_graham_scan::Point<Num_Type> newPoint = gpu_graham_scan::Point<int> ::operator +(partner);
      if (!comparePoints(current, partner)) {
        d_points[elt_of_interest] = current;
        d_points[partner_elt] = partner;
      } else {
        d_points[elt_of_interest] = partner;
        d_points[partner_elt] = current;
      }
    }
  }

  template <class Num_Type>
  __global__ void  BitonicSortPointsKernel(int n, gpu_graham_scan::Point<Num_Type>* d_points, int blocks_per_chunk, int chunk_size){
    int chunk_id = blockIdx.x / blocks_per_chunk; // could also use second BlockIdx.y
    int chunk_start = chunk_id * 2 * chunk_size;
    int chunk_end = chunk_start + chunk_size;
    
    int block_start = chunk_start + (blockIdx.x % blocks_per_chunk) * chunk_size/blocks_per_chunk;
    int block_end = block_start + chunk_size/blocks_per_chunk;
    
    int elt_of_interest = block_start + threadIdx.x;
    int step_size = chunk_size / 2;

    if (elt_of_interest < block_start + chunk_size / 2) {
      int partner_elt = elt_of_interest + step_size;
      gpu_graham_scan::Point<Num_Type> current = d_points[elt_of_interest];
      gpu_graham_scan::Point<Num_Type> partner = d_points[partner_elt];
      
      // gpu_graham_scan::Point<Num_Type> newPoint = gpu_graham_scan::Point<int> ::operator +(partner);
      if (!comparePoints(current, partner)) {
        d_points[elt_of_interest] = current;
        d_points[partner_elt] = partner;
      } else {
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

  // switch when chunk count => threads hits 256
  for (size_t i = 2, j = i; i <= upper_bound; i *= 2, j = i) {
    size_t chunks = (n_points + j - 1) / j;
    size_t chunk_size = j >> 1;

    int blocks_per_chunk = (chunk_size / kMaxThreads) + 1;


    // each thread gets its own chunk
    // each block has multiple chunks
    BuildBitonic<<<dim3(blocks_per_chunk*chunks), dim3(kMaxThreads)>>>(n_points, d_points, blocks_per_chunk, chunk_size);

    // wait for build to finish
    cudaErrorCheck(cudaDeviceSynchronize());
    j >>= 1; // j is the size of a chunk in this step
    while (j > 1) {
      int blocks_per_chunk = (j / kMaxThreads) + 1;
      // sort bionic
      BitonicSortPointsKernel<<<dim3(blocks_per_chunk*chunks), dim3(kMaxThreads)>>>(n_points, d_points, blocks_per_chunk, chunk_size);
      j >>= 1;
    }
  }

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