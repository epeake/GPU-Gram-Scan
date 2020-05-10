#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"
#include "gpu_graham_scan_test.h"

int kRuns = 3;
int kPoints = 4;

int main() {
  // std::vector<gpu_graham_scan::Point<double> > serial_output;
  // double serial_min_time = gpu_graham_scan_test::Benchmark(
  //     kRuns, gpu_graham_scan_test::SolveSerial<double>, kPoints, serial_output);
  // printf("[Graham-Scan serial]:\t\t%.3f ms\t%.3fX speedup\n",
  //        serial_min_time * 1000, 1.);

  // std::vector<gpu_graham_scan::Point<double> > parallel_output;
  // double parallel_min_time = gpu_graham_scan_test::Benchmark(
  //     kRuns, gpu_graham_scan_test::SolveParallel<double>, kPoints, parallel_output);
  // printf("[Graham-Scan parallel]:\t\t%.3f ms\t%.3fX speedup\n",
  //        parallel_min_time * 1000, serial_min_time / parallel_min_time);
  // if (!gpu_graham_scan_test::ValidateSolution(serial_output, parallel_output)) {
  //   std::cout << "Solution did not match serial implementation \n";
  // }

  gpu_graham_scan::GrahamScanSerial<double> test_serial(kPoints);
  std::sort(test_serial.points_.begin(), test_serial.points_.end());

  gpu_graham_scan::GrahamScanSerial<double> test_parallel(kPoints);
  gpu_graham_scan::BitonicSortPoints(test_parallel.points_);

  gpu_graham_scan::Point<double> p1(-73692.442452, -8269.973595);
  gpu_graham_scan::Point<double> p2(86938.579245, 3883.274405);
  std::cout << "test comp. " << (p1 < p2) << "\n"; 

  std::cout << "serial sort size: " << test_serial.points_.size() << '\n';
  std::cout << "parallel sort size: " << test_serial.points_.size() << '\n';

  std::cout << "p0_: " << test_serial.p0_.x_ << " , " << test_serial.p0_.y_ << "\n";
  std::cout << "serial sort: \n";
  for(int i =0; i<test_serial.points_.size();i++){
    gpu_graham_scan::Point<double> cp = test_serial.points_[i];
    std::cout << "point i: " << cp.x_ << " , " << cp.y_ << "\n";
  }



  std::cout << "\n parallel sort: \n";
  for(int i =0; i<test_parallel.points_.size();i++){
    gpu_graham_scan::Point<double> cp = test_parallel.points_[i];
    std::cout << "point "<< i << ": " << cp.x_ << " , " << cp.y_ << "\n";
  }


  // gpu_graham_scan::GrahamScanSerial<int> lol(kPoints);
  // gpu_graham_scan::BitonicSortPoints(lol.points_);
}
