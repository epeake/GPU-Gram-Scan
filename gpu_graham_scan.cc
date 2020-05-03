#include "gpu_graham_scan.h"

using namespace std;
using namespace gpu_graham_scan;

// bool operator<(const Point& a, const Point& b){
//     return a.p0_angle < b.p0_angle;
// }

// void GrahamScanSerial::SetP0(Point<Num_Type> p0){
//     Point<Num_Type> currentPoint;
//     for(int i=0; i<points_.size(); i++){
//         currentPoint = points_[i];
//         std::cout << "x, y: " << currentPoint.x << " " << currentPoint.y <<
//         "\n"; currentPoint.p0_angle = PolarAngle(p0, currentPoint);
//     }

//     std::sort(points_.begin(), points_.end());
// }