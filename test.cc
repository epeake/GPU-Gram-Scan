#include <iostream>

#include "CycleTimer.h"
#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

int kRuns = 3;

/*
 * find convex hull for points in input
 */
template <class Num_Type>
std::vector<Point<Num_Type> > SolveSerial(GrahamScanSerial<Num_Type>* input) {
  cout << "Solving serial\n";
  input->GetHull();
  return input->hull_;
}

template <class Num_Type>
bool ValidateSolution(std::vector<Point<Num_Type> >& soln1,
                      std::vector<Point<Num_Type> >& soln2) {
  cout << "Validating Solution \n";
  cout << "soln1 size: " << soln1.size();
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
  }
  return true;
}

template <class Fn, class Num_Type>
double Benchmark(int num_runs, Fn&& fn, const char* filename,
                 std::vector<Point<Num_Type> >& hull) {
  double min_time = std::numeric_limits<double>::max();
  for (int i = 0; i < num_runs; i++) {
    GrahamScanSerial<int> input(filename);
    double start_time = CycleTimer::currentSeconds();
    std::vector<Point<Num_Type> > temp = fn(&input);
    hull = temp;
    cout << "benchmark hull size: " << hull.size() << "\n";
    double end_time = CycleTimer::currentSeconds();
    min_time = std::min(min_time, end_time - start_time);
  }
  return min_time;
}

int main() {
  // Point<int> p1;
  // p1.x = 1;
  // p1.y = 2;

  GrahamScanSerial<int> test("test-data/test1.in");
  cout << "constructed \n";

  std::vector<Point<int> > output;
  // output = std::move(output);
  // std::vector<int>&& output_vec = output;
  // output = SolveSerial(&test);
  // cout << "solve serial:" << output.size() << "\n";

  double min_time =
      Benchmark(kRuns, SolveSerial<int>, "test-data/test1.in", output);
  cout << "min time was: " << min_time << " seconds\n";
  cout << "validate solution\n";
  cout << output.size() << "\n";
  cout << ValidateSolution(output, output);

  // Benchmark(kRuns, Run)
  // cout << "test min point: " << test.GetMinYPoint().x << ' ' <<
  // test.GetMinYPoint().y << '\n'; cout << "true min point: -1 -1 \n"; cout <<
  // "num points: " << test.points_.size() << "\n"; Point<int> p2; p2.x = 1;
  // p2.y = 1;

  // cout << "polar angle: p2 -> p1 " << test.PolarAngle(p2, p1) <<  "\n";

  // Point<int> p3;
  // p3.x = 2;
  // p3.y = 1;

  // cout << "non left turn: " << test.NonLeftTurn(p1, p3, p2) << "\n";
  // cout << "non left turn: " << test.NonLeftTurn(p1, p2, p3) << "\n";

  // test.CenterP0(test.GetMinYPoint());

  // cout << "begin for loop\n";
  // for (int i = 0; i < test.points_.size(); i++) {
  //   cout << i << "\n";
  //   Point<int> current_point = test.points_[i];
  //   std::cout << "x, y: " << current_point.x << " " << current_point.y <<
  //   "\n";
  // }
}
