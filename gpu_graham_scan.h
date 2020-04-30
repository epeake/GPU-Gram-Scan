#pragma once

#ifndef _GPU_Graham_Scan_
#define _GPU_Graham_Scan_

#include <errno.h>
#include <fstream>
#include <string>
#include <vector>
#include <cmath> // trig functions, pow

using std::string;
using std::vector;
using std::ifstream;

namespace gpu_graham_scan {

const float kPI = 4 * atan(1);

/*
 * Cartesian Coordinate Point
 */
template <class Num_Type> struct Point {
  Num_Type x;
  Num_Type y;
};

/*
 * Used to read in a file of Point to be stored
 * as a vector.
 */
template <class Num_Type> class GrahamScanSerial {
  public:

    /*
     * Constructor reads through the file, populating points_ and idx_min_y_
     */
    GrahamScanSerial(const char * filename) : filename_(filename) {
      string curr_line;
      ifstream infile;
      int idx = 0;

      infile.open(filename_);
      if (errno != 0) {
        perror("infile.open");
        exit(EXIT_FAILURE);
      }
      
      // get number of points
      getline(infile, curr_line);
      if (infile.fail()) {
        perror("getline");
        exit(EXIT_FAILURE);
      }

      points_ = new Point<Num_Type>[stoi(curr_line)];

      // process each line individually of the file
      while (!infile.eof()) {
        getline(infile, curr_line);
        if (infile.fail()) {
          perror("getline");
          exit(EXIT_FAILURE);
        }
        int comma = curr_line.find(',');
        string first_num = curr_line.substr(0, comma);
        string second_num = curr_line.substr(comma + 1, curr_line.length());

        Point<Num_Type> current_point;
        current_point.x = (Num_Type) stod(first_num);
        current_point.y = (Num_Type) stod(second_num);

        // update the current minumim point's index
        if (idx == 0
            || (current_point.y == GetMinYPoint().y && current_point.x < GetMinYPoint().x)
            || current_point.y < GetMinYPoint().y) {
          idx_min_y_ = idx;
        }

        points_[idx] = current_point;
        idx++;
      }

      infile.close();
      if (infile.fail()) {
          perror("infile.close");
          exit(EXIT_FAILURE);
      }
    };

    ~GrahamScanSerial() { delete[] points_; }

    /*
     * filename of points to be read in
     */
    string filename_;

    /*
     * index of the leftmost minimum y-coordinate point in points_
     */
    int idx_min_y_;

    /*
     * all of our points from the file
     */
    Point<Num_Type> * points_;

    
    Point<Num_Type> GetMinYPoint() {
      return points_[idx_min_y_];
    };


  /* 
   * calculate polar angles between two points
   * parallel to x-axis is 0
   * 
   * args: Point p0 - translated to the origin
   *       Point p1 - another point in reference to p0
   * returns: the polar angle of p1 where p0 is the origin (radians)
   */
  float PolarAngle(Point<Num_Type> p0, Point<Num_Type> p1) const {
    float x_diff = p1.x - p0.x;
    float y_diff = p1.y - p0.y;

    float hypotenuse = hypotf(x_diff, y_diff);
    float raw_angle = acos(x_diff/hypotenuse); // use cosine so the function is always defined
    if(y_diff < 0){
      return 2*kPI - raw_angle;
    } else {
      return raw_angle;
    }
  }


  /* 
   * does the ordering self, p1, p2 create a non-left turn
   * 
   * args: three Points that form a turn
   * returns: if p0->p2 is a non-left turn relative to p0->p1
   */
  bool NonLeftTurn(Point<Num_Type> p0, Point<Num_Type> p1, Point<Num_Type> p2) const {
    float cross_product = (p2.x-p0.x)*(p1.y-p0.y) - (p1.x-p0.x)*(p2.y-p0.y);
    return cross_product > 0;
  }
  
  private:
    GrahamScanSerial(void);
};

} // gpu_graham_scan

#endif // _GPU_Graham_Scan_
