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
__global__ void BuildBitonic(size_t n_points,
                             gpu_graham_scan::Point<Num_Type>* d_points,
                             size_t chunk_size) {
  size_t first = threadIdx.x + (blockIdx.x * (chunk_size * blockDim.x));
  size_t last = first + chunk_size - 1;

  while (first < last) {
    if (comparePoints(d_points[first], d_points[last])) {
      gpu_graham_scan::Point<Num_Type> tmp = d_points[last];
      d_points[last] = d_points[first];
      d_points[first] = tmp;
    }
    first++;
    last--;
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

    // each thread gets its own chunk
    // each block has multiple chunks
    BuildBitonic<<<(chunks + kChunksPerBlock - 1) / kChunksPerBlock,
                   kChunksPerBlock>>>(n_points, d_points, chunk_size);

    // wait for build to finish
    cudaErrorCheck(cudaDeviceSynchronize());
    j >>= 1;
    while (j > 1) {
      // sort bionic
      // BitonicSortPointsKernel<<<dim3(BX, BY), dim3(TX, TY)>>>();
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

template void __global__ BuildBitonic(size_t n_points,
                                      gpu_graham_scan::Point<int>* d_points,
                                      size_t chunk_size);

template void __global__ BuildBitonic(size_t n_points,
                                      gpu_graham_scan::Point<double>* d_points,
                                      size_t chunk_size);

template __device__ bool comparePoints(const gpu_graham_scan::Point<int> p1,
                                       const gpu_graham_scan::Point<int> p2);

template __device__ bool comparePoints(const gpu_graham_scan::Point<double> p1,
                                       const gpu_graham_scan::Point<double> p2);
