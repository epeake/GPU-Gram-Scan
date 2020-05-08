#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

int kRuns = 50;
int kPoints = 10000;

/*
 * find convex hull for points in input
 * serial implementation
 */
template <class Num_Type>
std::vector<Point<Num_Type> > SolveSerial(GrahamScanSerial<Num_Type>* input) {
  input->GetHullSerial();
  return input->hull_;
}

/*
 * find convex hull for points in input
 * parallel implementation
 */
template <class Num_Type>
std::vector<Point<Num_Type> > SolveParallel(GrahamScanSerial<Num_Type>* input) {
  input->GetHullParallel();
  return input->hull_;
}

template <class Num_Type>
bool ValidateSolution(std::vector<Point<Num_Type> >& soln1,
                      std::vector<Point<Num_Type> >& soln2) {
  if (soln1.size() != soln2.size()) {
    fprintf(stderr, "Hulls have different numbers of elements");
    return false;
  }

  // could make this O(n) with unordered_map, but don't want to make hash
  // function => we have an easy O(nlog(n))
  std::sort(soln1.begin(), soln1.end());
  std::sort(soln2.begin(), soln2.end());

  int i = 0;
  while (i < soln1.size()) {
    if (soln1[i] != soln2[i]) {
      return false;
    }
    i++;
  }
  return true;
}

template <class Fn, class Num_Type>
double Benchmark(int num_runs, Fn&& fn, int n,
                 std::vector<Point<Num_Type> >& hull) {
  double min_time = std::numeric_limits<double>::max();
  for (int i = 0; i < num_runs; i++) {
    GrahamScanSerial<int> input(n);
    double start_time = CycleTimer::currentSeconds();
    std::vector<Point<Num_Type> > temp = fn(&input);
    hull = temp;
    double end_time = CycleTimer::currentSeconds();
    min_time = std::min(min_time, end_time - start_time);
  }
  return min_time;
}

int main() {
  GrahamScanSerial<int> test1(5);
  GrahamScanSerial<int> test2(5);

  std::vector<Point<int> > serial_output;
  double serial_min_time =
      Benchmark(kRuns, SolveSerial<int>, kPoints, serial_output);
  printf("[Graham-Scan serial]:\t\t%.3f ms\t%.3fX speedup\n",
         serial_min_time * 1000, 1.);

  std::vector<Point<int> > parallel_output;
  double parallel_min_time =
      Benchmark(kRuns, SolveSerial<int>, kPoints, parallel_output);
  printf("[Graham-Scan parallel]:\t\t%.3f ms\t%.3fX speedup\n",
         parallel_min_time * 1000, serial_min_time / parallel_min_time);
  if (!ValidateSolution(serial_output, parallel_output)) {
    cout << "Solution did not match serial implementation";
  }
}
