#include <iostream>
#include <vector>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"
#include "gpu_graham_scan_test.h"

int kRuns = 5;
unsigned long kPoints = 1000000;
bool test_int = true;
bool test_float = true;
bool test_double = true;

int main() {
  if (test_int) {
    std::cout << "Testing type == int:\n";
    std::vector<gpu_graham_scan::Point<int>> serial_output;
    double serial_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveSerial<int>, kPoints, serial_output);
    printf("[Graham-Scan serial %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, serial_min_time * 1000, 1.);

    std::vector<gpu_graham_scan::Point<int>> parallel_output;
    double parallel_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveParallel<int>, kPoints,
        parallel_output);
    printf("[Graham-Scan parallel %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, parallel_min_time * 1000,
           serial_min_time / parallel_min_time);
    if (!gpu_graham_scan_test::ValidateSolution(serial_output,
                                                parallel_output)) {
      std::cout << "Solution did not match serial implementation\n";
    }
  }

  if (test_float) {
    std::cout << "\nTesting type == float:\n";
    std::vector<gpu_graham_scan::Point<float>> serial_output;
    double serial_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveSerial<float>, kPoints,
        serial_output);
    printf("[Graham-Scan serial %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, serial_min_time * 1000, 1.);

    std::vector<gpu_graham_scan::Point<float>> parallel_output;
    double parallel_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveParallel<float>, kPoints,
        parallel_output);
    printf("[Graham-Scan parallel %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, parallel_min_time * 1000,
           serial_min_time / parallel_min_time);
    if (!gpu_graham_scan_test::ValidateSolution(serial_output,
                                                parallel_output)) {
      std::cout << "Solution did not match serial implementation\n";
    }
  }

  if (test_double) {
    std::cout << "\nTesting type == double:\n";
    std::vector<gpu_graham_scan::Point<double>> serial_output;
    double serial_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveSerial<double>, kPoints,
        serial_output);
    printf("[Graham-Scan serial %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, serial_min_time * 1000, 1.);

    std::vector<gpu_graham_scan::Point<double>> parallel_output;
    double parallel_min_time = gpu_graham_scan_test::Benchmark(
        kRuns, gpu_graham_scan_test::SolveParallel<double>, kPoints,
        parallel_output);
    printf("[Graham-Scan parallel %lu points]:\t\t%.3f ms\t%.3fX speedup\n",
           kPoints, parallel_min_time * 1000,
           serial_min_time / parallel_min_time);
    if (!gpu_graham_scan_test::ValidateSolution(serial_output,
                                                parallel_output)) {
      std::cout << "Solution did not match serial implementation\n";
    }
  }
}
