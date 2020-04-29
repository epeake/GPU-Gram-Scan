#pragma once

#ifndef _GPU_Gram_Scan_
#define _GPU_Gram_Scan_

#include <errno.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::ifstream;

namespace gpu_gram_scan {

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
template <class Num_Type> class PointFileReader {
  public:

    /*
     * Constructor reads through the file, populating points_ and idx_min_y_
     */
    PointFileReader(const char * filename) : filename_(filename) {
      string curr_line;
      ifstream infile;
      int idx = 0;

      infile.open(filename_);
      if (errno != 0) {
        perror("infile.open");
        exit(EXIT_FAILURE);
      }
      
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

        Point<int> current_point;
        current_point.x = stoi(first_num);
        current_point.y = stoi(second_num);

        // update the current minumim point's index
        if (idx == 0
            || (current_point.y == GetMinYPoint().y && current_point.x < GetMinYPoint().x)
            || current_point.y < GetMinYPoint().y) {
          idx_min_y_ = idx;
        }

        points_.push_back(current_point);
        idx++;
      }

      infile.close();
      if (infile.fail()) {
          perror("infile.close");
          exit(EXIT_FAILURE);
      }
    };

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
    vector< Point<Num_Type> > points_;

    
    Point<Num_Type> GetMinYPoint() {
      return points_[idx_min_y_];
    };
  
  private:
    PointFileReader(void);
};

} // gpu_gram_scan

#endif // _GPU_Gram_Scan_
