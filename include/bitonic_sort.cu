#include <vector>

#include "cuda-util.h"
#include "gpu_graham_scan.h"

template <class Num_Type>
__global__ void BitonicSortPointsKernel() {}

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

  // // blocks in X and Y directions
  // const int BX = ((image.width_ - 1) / TX) + 1;
  // const int BY = ((image.height_ - 1) / TY) + 1;

  // // Compute the width and height of each pixel in normalized [0,1]
  // coordinates const float x_width = 1.f / image.width_; const float y_width
  // = 1.f / image.height_;

  // BitonicSortPointsKernel<<<dim3(BX, BY), dim3(TX, TY)>>>(
  //     n_circles, d_circles_position, d_circles_radius, d_circles_color,
  //     image.width_, image.height_, x_width, y_width, d_image_data);

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
