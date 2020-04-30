#include <iostream>
#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

int main() {
  Point<int> p1;
  p1.x = 1;
  p1.y = 2;

  PointFileReader<double> test("test-data/test1.in");
  cout << test.GetMinYPoint().x << ' ' << test.GetMinYPoint().y << '\n';

  Point<int> p2;
  p2.x = 1;
  p2.y = 1;

  cout << "polar angle: p2 -> p1 " << p2.PolarAngle(p1) <<  "\n";

  Point<int> p3;
  p2.x = 2;
  p2.y = 1;
  
  cout << "non left turn: " << p1.NonLeftTurn(p3,p2) << "\n";
  cout << "non left turn: " << p1.NonLeftTurn(p2,p3) << "\n";
}
