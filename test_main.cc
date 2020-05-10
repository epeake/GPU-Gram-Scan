#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"
#include "gpu_graham_scan_test.h"

int kRuns = 3;
int kPoints = 100000;

int main() {
  std::vector<gpu_graham_scan::Point<double> > serial_output;
  double serial_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveSerial<double>, kPoints, serial_output);
  printf("[Graham-Scan serial]:\t\t%.3f ms\t%.3fX speedup\n",
         serial_min_time * 1000, 1.);

  std::vector<gpu_graham_scan::Point<double> > parallel_output;
  double parallel_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveParallel<double>, kPoints, parallel_output);
  printf("[Graham-Scan parallel]:\t\t%.3f ms\t%.3fX speedup\n",
         parallel_min_time * 1000, serial_min_time / parallel_min_time);
  if (!gpu_graham_scan_test::ValidateSolution(serial_output, parallel_output)) {
    std::cout << "Solution did not match serial implementation \n";
  }

  gpu_graham_scan::GrahamScanSerial<int> lol(kPoints);
  gpu_graham_scan::BitonicSortPoints(lol.points_);
}
