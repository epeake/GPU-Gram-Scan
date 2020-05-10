#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"
#include "gpu_graham_scan_test.h"

int kRuns = 50;
int kPoints = 5;

int main() {
  std::vector<gpu_graham_scan::Point<int> > serial_output;
  double serial_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveSerial<int>, kPoints, serial_output);
  printf("[Graham-Scan serial]:\t\t%.3f ms\t%.3fX speedup\n",
         serial_min_time * 1000, 1.);

  std::vector<gpu_graham_scan::Point<int> > parallel_output;
  double parallel_min_time = gpu_graham_scan_test::Benchmark(
      kRuns, gpu_graham_scan_test::SolveSerial<int>, kPoints, parallel_output);
  printf("[Graham-Scan parallel]:\t\t%.3f ms\t%.3fX speedup\n",
         parallel_min_time * 1000, serial_min_time / parallel_min_time);
  if (!gpu_graham_scan_test::ValidateSolution(serial_output, parallel_output)) {
    std::cout << "Solution did not match serial implementation";
  }

  gpu_graham_scan::GrahamScanSerial<int> lol(kPoints);
  gpu_graham_scan::BitonicSortPoints(lol.points_);

  gpu_graham_scan::GrahamScanSerial<int> lol2(kPoints);
  std::cout << "correct before: \n\n";
  std::cout << "p0: " << lol2.points_[0].x_ << " " << lol2.points_[0].y_
            << '\n';
  std::cout << "p1: " << lol2.points_[1].x_ << " " << lol2.points_[1].y_
            << '\n';
  std::cout << "p2: " << lol2.points_[2].x_ << " " << lol2.points_[2].y_
            << '\n';
  std::cout << "p3: " << lol2.points_[3].x_ << " " << lol2.points_[3].y_
            << '\n';
  std::cout << "p4: " << lol2.points_[4].x_ << " " << lol2.points_[4].y_
            << '\n';
  std::sort(lol2.points_.begin(), lol2.points_.end());
  std::cout << "correct after: \n";
  std::cout << "p0: " << lol2.points_[0].x_ << " " << lol2.points_[0].y_
            << '\n';
  std::cout << "p1: " << lol2.points_[1].x_ << " " << lol2.points_[1].y_
            << '\n';
  std::cout << "p2: " << lol2.points_[2].x_ << " " << lol2.points_[2].y_
            << '\n';
  std::cout << "p3: " << lol2.points_[3].x_ << " " << lol2.points_[3].y_
            << '\n';
  std::cout << "p4: " << lol2.points_[4].x_ << " " << lol2.points_[4].y_
            << '\n';

  gpu_graham_scan::Point<int> p1(-4, -4);
  gpu_graham_scan::Point<int> p2(4, 0);

  std::cout << "should be true: " << (p2 < p1) << '\n';
  std::cout << "should be true: " << gpu_graham_scan::comparePoints(p2, p1)
            << '\n';
}
