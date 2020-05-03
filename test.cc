#include <iostream>
#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

int main() {
  // Point<int> p1;
  // p1.x = 1;
  // p1.y = 2;

  GrahamScanSerial<int> test("test-data/test1.in");
  // cout << "test min point: " << test.GetMinYPoint().x << ' ' << test.GetMinYPoint().y << '\n';
  // cout << "true min point: -1 -1 \n";
  // cout << "num points: " << test.points_.size() << "\n";
  // Point<int> p2;
  // p2.x = 1;
  // p2.y = 1;

  // cout << "polar angle: p2 -> p1 " << test.PolarAngle(p2, p1) <<  "\n";

  // Point<int> p3;
  // p3.x = 2;
  // p3.y = 1;
  
  // cout << "non left turn: " << test.NonLeftTurn(p1, p3, p2) << "\n";
  // cout << "non left turn: " << test.NonLeftTurn(p1, p2, p3) << "\n";

  // test.CenterP0(test.GetMinYPoint());

  for(int i=0; i<test.points_.size(); i++){
    cout << i << "\n";
    Point<int> current_point = test.points_[i];
    std::cout << "x, y: " << current_point.x << " " << current_point.y <<"\n";
  }
}
