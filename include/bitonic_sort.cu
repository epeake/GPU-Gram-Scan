#include <iostream>
#include <vector>

#include "cuda-util.h"
#include "gpu_graham_scan.h"

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
__global__ void BuildBitonicKernel(size_t n_points,
                                   gpu_graham_scan::Point<Num_Type>* d_points,
                                   size_t threads_per_chunk, size_t chunk_len) {
  // should have 1/2 as many total threads as points
  size_t true_idx = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t chunk_offset = (true_idx / threads_per_chunk) * chunk_len;
  size_t thread_offset = true_idx % threads_per_chunk;
  size_t first = thread_offset + chunk_offset;
  size_t last = (chunk_offset + chunk_len - 1) - thread_offset;
  if (last < n_points && comparePoints(d_points[last], d_points[first])) {
    // printf("bfirst: %lu last: %lu    p1: %d %d  p2: %d %d \n", first, last,
    //        d_points[first].x_, d_points[first].y_, d_points[last].x_,
    //        d_points[last].y_);
    gpu_graham_scan::Point<Num_Type> tmp = d_points[last];
    d_points[last] = d_points[first];
    d_points[first] = tmp;
  } else if (last < n_points &&
             !comparePoints(d_points[last], d_points[first])) {
    // printf("no bfirst: %lu last: %lu    p1: %d %d  p2: %d %d \n", first,
    // last,
    //        d_points[first].x_, d_points[first].y_, d_points[last].x_,
    //        d_points[last].y_);
  }
}

template <class Num_Type>
__global__ void BitonicSortKernel(size_t n_points,
                                  gpu_graham_scan::Point<Num_Type>* d_points,
                                  size_t threads_per_chunk, size_t chunk_len) {
  // should have 1/2 as many total threads as points
  size_t true_idx = threadIdx.x + (blockIdx.x * blockDim.x);
  size_t chunk_offset = (true_idx / threads_per_chunk) * chunk_len;
  size_t thread_offset = true_idx % threads_per_chunk;
  size_t first = thread_offset + chunk_offset;
  size_t last = first + threads_per_chunk;
  if (last < n_points && comparePoints(d_points[last], d_points[first])) {
    // printf("sfirst: %lu last: %lu    p1: %d %d  p2: %d %d \n", first, last,
    //        d_points[first].x_, d_points[first].y_, d_points[last].x_,
    //        d_points[last].y_);
    gpu_graham_scan::Point<Num_Type> tmp = d_points[last];
    d_points[last] = d_points[first];
    d_points[first] = tmp;
  } else if (last < n_points &&
             !comparePoints(d_points[last], d_points[first])) {
    // printf("no sfirst: %lu last: %lu    p1: %d %d  p2: %d %d \n", first,
    // last,
    //        d_points[first].x_, d_points[first].y_, d_points[last].x_,
    //        d_points[last].y_);
  }
}

template <class Num_Type>
void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<Num_Type>>& points) {
  const uint kThreadsPerBlock = 1;  // Max kThreadsPerBlock = 1024;
  size_t n_points = points.size();
  size_t total_threads = (n_points + 1) / 2;

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
  std::cout << "running...\n";
  for (size_t i = 2, j = i; i <= upper_bound; i *= 2, j = i) {
    size_t threads_per_chunk = j >> 1;
    BuildBitonicKernel<<<(total_threads + kThreadsPerBlock - 1) /
                             kThreadsPerBlock,
                         kThreadsPerBlock>>>(n_points, d_points,
                                             threads_per_chunk, j);

    // wait for build to finish
    cudaErrorCheck(cudaDeviceSynchronize());
    j >>= 1;
    while (j > 1) {
      threads_per_chunk = j >> 1;
      BitonicSortKernel<<<(total_threads + kThreadsPerBlock - 1) /
                              kThreadsPerBlock,
                          kThreadsPerBlock>>>(n_points, d_points,
                                              threads_per_chunk, j);
      cudaErrorCheck(cudaDeviceSynchronize());
      j >>= 1;
    }
  }
  std::cout << "\n\n";
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
    std::vector<gpu_graham_scan::Point<int>>& points);

template void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<double>>& points);

template void __global__
BuildBitonicKernel(size_t n_points, gpu_graham_scan::Point<int>* d_points,
                   size_t threads_per_chunk, size_t chunk_len);

template void __global__
BuildBitonicKernel(size_t n_points, gpu_graham_scan::Point<double>* d_points,
                   size_t threads_per_chunk, size_t chunk_len);

template __global__ void BitonicSortKernel(
    size_t n_points, gpu_graham_scan::Point<int>* d_points,
    size_t threads_per_chunk, size_t chunk_len);

template __global__ void BitonicSortKernel(
    size_t n_points, gpu_graham_scan::Point<double>* d_points,
    size_t threads_per_chunk, size_t chunk_len);

template __device__ bool comparePoints(const gpu_graham_scan::Point<int> p1,
                                       const gpu_graham_scan::Point<int> p2);

template __device__ bool comparePoints(const gpu_graham_scan::Point<double> p1,
                                       const gpu_graham_scan::Point<double> p2);
