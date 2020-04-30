#include <iostream>
#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

int main() {
  Point<int> p1;
  p1.x = 1;
  p1.y = 2;

  GrahamScanSerial<int> test("test-data/test1.in");
  cout << test.GetMinYPoint().x << ' ' << test.GetMinYPoint().y << '\n';

  Point<int> p2;
  p2.x = 1;
  p2.y = 1;

  cout << "polar angle: p2 -> p1 " << test.PolarAngle(p2, p1) <<  "\n";

  Point<int> p3;
  p3.x = 2;
  p3.y = 1;
  
  cout << "non left turn: " << test.NonLeftTurn(p1, p3, p2) << "\n";
  cout << "non left turn: " << test.NonLeftTurn(p1, p2, p3) << "\n";
}
