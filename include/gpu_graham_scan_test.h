#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"

namespace gpu_graham_scan_test {

/*
 * find convex hull for points in input
 * serial implementation
 */
template <class Num_Type>
std::vector<gpu_graham_scan::Point<Num_Type> > SolveSerial(
    gpu_graham_scan::GrahamScanSerial<Num_Type>* input) {
  input->GetHullSerial();
  return input->hull_;
}

/*
 * find convex hull for points in input
 * parallel implementation
 */
template <class Num_Type>
std::vector<gpu_graham_scan::Point<Num_Type> > SolveParallel(
    gpu_graham_scan::GrahamScanSerial<Num_Type>* input) {
  input->GetHullParallel();
  return input->hull_;
}

template <class Num_Type>
bool ValidateSolution(std::vector<gpu_graham_scan::Point<Num_Type> >& soln1,
                      std::vector<gpu_graham_scan::Point<Num_Type> >& soln2) {
  if (soln1.size() != soln2.size()) {
    fprintf(stderr, "Hulls have different numbers of elements");
    return false;
  }

  // could make this O(n) with unordered_map, but don't want to make hash
  // function => we have an easy O(nlog(n))
  std::sort(soln1.begin(), soln1.end());
  std::sort(soln2.begin(), soln2.end());

  size_t i = 0;
  while (i < soln1.size()) {
    if (soln1[i] != soln2[i]) {
      return false;
    }
    i++;
  }
  return true;
}

template <class Fn, class Num_Type, class In_Type>
double Benchmark(int num_runs, Fn&& fn, In_Type n,
                 std::vector<gpu_graham_scan::Point<Num_Type> >& hull) {
  double min_time = std::numeric_limits<double>::max();
  for (int i = 0; i < num_runs; i++) {
    gpu_graham_scan::GrahamScanSerial<Num_Type> input(n);
    double start_time = CycleTimer::currentSeconds();
    std::vector<gpu_graham_scan::Point<Num_Type> > temp = fn(&input);
    hull = temp;
    double end_time = CycleTimer::currentSeconds();
    min_time = std::min(min_time, end_time - start_time);
  }
  return min_time;
}

}  // namespace gpu_graham_scan_test