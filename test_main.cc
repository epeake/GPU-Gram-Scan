#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"
#include "gpu_graham_scan_test.h"

int kRuns = 50;
int kPoints = 20000;

int main() {
  std::vector<gpu_graham_scan::Point<int>> serial_output;
  double serial_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveSerial<int>, kPoints, serial_output);
  printf("[Graham-Scan serial]:\t\t%.3f ms\t%.3fX speedup\n",
         serial_min_time * 1000, 1.);

  std::vector<gpu_graham_scan::Point<int>> parallel_output;
  double parallel_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveParallel<int>, kPoints,
      parallel_output);
  printf("[Graham-Scan parallel]:\t\t%.3f ms\t%.3fX speedup\n",
         parallel_min_time * 1000, serial_min_time / parallel_min_time);
  if (!gpu_graham_scan_test::ValidateSolution(serial_output, parallel_output)) {
    std::cout << "Solution did not match serial implementation";
  }

  std::vector<int> help;
  for (int pts = 10; pts < 30 + 1; pts++) {
    gpu_graham_scan::GrahamScanSerial<int> lol(pts);
    std::vector<gpu_graham_scan::Point<int>> lol2;
    for (auto& i : lol.points_) {
      lol2.push_back(i);
    }

    gpu_graham_scan::BitonicSortPoints(lol.points_.data() + 1,
                                       lol.points_.size() - 1);

    std::sort(lol2.begin() + 1, lol2.end());
    bool is_same = true;
    for (size_t i = 0; i < lol2.size(); i++) {
      if (lol2[i].x_ != lol.points_[i].x_ || lol2[i].y_ != lol.points_[i].y_) {
        is_same = false;
      }
    }

    if (!is_same) help.push_back(pts);
  }

  for (auto& xxx : help) {
    std::cout << "bad: " << xxx << "\n";
  }
}
