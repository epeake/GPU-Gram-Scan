#include <iostream>
#include <vector>

#include "cuda-util.h"
#include "gpu_graham_scan.h"

const uint kMaxThreads = 1024;
const uint kTotalSMs = 46;
const uint kMaxThreadsSpan = kMaxThreads * kTotalSMs * 2;

template <class Num_Type>
__global__ void BuildBitonic(size_t start_size) {
  while (start_size > 1) {
    start_size >>= 1;
  }
}

template <class Num_Type>
void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<Num_Type>> points) {
  size_t n_points = points.size();
  Num_Type xs[n_points];
  Num_Type ys[n_points];

  for (size_t i = 0; i < n_points; i++) {
    gpu_graham_scan::Point<Num_Type> curr = points[i];
    xs[i] = curr.x_;
    ys[i] = curr.y_;
  }

  // Allocate device data
  Num_Type* d_xs;
  Num_Type* d_ys;

  cudaErrorCheck(cudaMalloc(&d_xs, n_points * sizeof(Num_Type)));
  cudaErrorCheck(cudaMalloc(&d_ys, n_points * sizeof(Num_Type)));

  // points to device
  cudaErrorCheck(cudaMemcpy(d_xs, xs, n_points * sizeof(Num_Type),
                            cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(d_ys, ys, n_points * sizeof(Num_Type),
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

  for (size_t i = 2, j = i; i <= upper_bound; i *= 2, j = i) {
    std::cout << j << '\n';
    size_t chunks;
    size_t threads_per_chunk;
    if (j > kMaxThreadsSpan) {
      chunks = 1;
      threads_per_chunk = 1;
    } else {
      chunks = (n_points + j - 1) / j;
      threads_per_chunk = j >> 1;
    }

    // BuildBitonic<<<>>>;
    j >>= 1;
    while (j > 1) {
      std::cout << j << '\n';
      // sort bionic
      // BitonicSortPointsKernel<<<dim3(BX, BY), dim3(TX, TY)>>>();
      j >>= 1;
    }
  }

  // Copy points back to host
  cudaErrorCheck(cudaMemcpy(xs, d_xs, n_points * sizeof(Num_Type),
                            cudaMemcpyDeviceToHost));
  cudaErrorCheck(cudaMemcpy(ys, d_ys, n_points * sizeof(Num_Type),
                            cudaMemcpyDeviceToHost));

  // Cleanup device data
  cudaErrorCheck(cudaFree(d_xs));
  cudaErrorCheck(cudaFree(d_ys));
}

template void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<int>> points);

template void gpu_graham_scan::BitonicSortPoints(
    std::vector<gpu_graham_scan::Point<double>> points);
