#include <iostream>
#include "gpu_gram_scan.h"

using namespace std;
using namespace gpu_gram_scan;

int main() {
  Point<int> p1;
  p1.x = 1;
  p1.y = 2;

  PointFileReader<int> test("test-data/test1.in");
  cout << test.GetMinYPoint().x << ' ' << test.GetMinYPoint().y << '\n';

  Point<int> p2;
  p2.x = 1;
  p2.y = 1;

  cout <<p2.PolarAngle(p1) << "polar angle: p2 -> p1 \n";
}
