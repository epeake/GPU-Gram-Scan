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
}
