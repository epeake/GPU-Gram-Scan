#include <algorithm>

#include "CycleTimer.h"
#pragma once

#ifndef _GPU_Graham_Scan_
#define _GPU_Graham_Scan_

/*
 * prints helpful error diagnostics
 */
#define GPU_GS_PRINT_ERR(message)                                          \
  fprintf(stderr, "Error: function %s, file %s, line %d.\n%s\n", __func__, \
          __FILE__, __LINE__, message);
#define GPU_GS_PRINT_ERR_LOC()                                         \
  fprintf(stderr, "Error: function %s, file %s, line %d.\n", __func__, \
          __FILE__, __LINE__);

#include <errno.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <random>
#include <stack>
#include <string>
#include <thread>
#include <vector>

namespace gpu_graham_scan {

/*
 * cartesian coordinate point
 */
template <class Num_Type>
struct Point {
  Num_Type x_;
  Num_Type y_;

  Point(Num_Type x = 0, Num_Type y = 0) : x_(x), y_(y) {}
};

/*
 * the directions we can turn in, used when seeing if two points make a
 * right, left, or no turn
 */
enum TurnDir { RIGHT, NONE, LEFT };

/*
 * subtract two Points from each other
 *
 * params:
 *  p1: our first point
 *  p2: our second point
 *
 * returns:
 *  Point<Num_Type> of the difference
 */
template <class Num_Type>
Point<Num_Type> operator-(const Point<Num_Type>& p1,
                          const Point<Num_Type>& p2) {
  return Point<Num_Type>(p1.x_ - p2.x_, p1.y_ - p2.y_);
}

/*
 * add two Points to each other
 *
 * params:
 *  p1: our first point
 *  p2: our second point
 *
 * returns:
 *  Point<Num_Type> of the sum
 */
template <class Num_Type>
Point<Num_Type> operator+(const Point<Num_Type>& p1,
                          const Point<Num_Type>& p2) {
  return Point<Num_Type>(p1.x_ + p2.x_, p1.y_ + p2.y_);
}

/*
 * determines whether p1 is to the right or left of p2
 *
 * params:
 *  p1: starting point
 *  p2: point whose location we are seeking relative to p1
 *
 * returns:
 *  TurnDir:
 *    RIGHT: p1 to the right of p2
 *    NONE:  p1 and p2 colinear
 *    LEFT:  p1 to the left of p2
 */
template <class Num_Type>
TurnDir GetTurnDir(const Point<Num_Type>& p1, const Point<Num_Type>& p2) {
  Num_Type x_product = XProduct(p1, p2);
  if (x_product > 0) {
    return RIGHT;
  }
  if (x_product == 0) {
    return NONE;
  }
  return LEFT;
}

/*
 * Comparator for two points used for sorting points based on polar angle.
 * If polar angles are the same, we sort in increasing order of squared
 * magnitude.
 */
template <class Num_Type>
bool operator<(const Point<Num_Type>& p1, const Point<Num_Type>& p2) {
  TurnDir dir = GetTurnDir(p1, p2);
  if (dir == NONE) {
    return SqrdMagnitude(p1) < SqrdMagnitude(p2);
  }
  return dir == RIGHT;
}

/*
 * Check to see if two points are equal
 */
template <class Num_Type>
bool operator!=(const Point<Num_Type>& p1, const Point<Num_Type>& p2) {
  return p1.x_ != p2.x_ || p1.y_ != p2.y_;
}

/*
 * calculate the cross product between two Points
 *
 * params:
 *  p1: our first point
 *  p2: our second point
 *
 * returns:
 *  float: our cross product
 */
template <class Num_Type>
Num_Type XProduct(const Point<Num_Type>& p1, const Point<Num_Type>& p2) {
  return (p1.x_ * p2.y_) - (p2.x_ * p1.y_);
}

/*
 * calculate the squared magnitude of a vector
 *
 * params:
 *  p1: our first point
 *  p2: our second point
 *
 * returns:
 *  Num_Type: our squared magnitude
 */
template <class Num_Type>
Num_Type SqrdMagnitude(const Point<Num_Type>& p) {
  return (p.x_ * p.x_) + (p.y_ * p.y_);
}

/*
 * GrahamScanSerial computes and stores our convex hull using points from a
 * specified file in serial
 */
template <class Num_Type>
class GrahamScanSerial {
 public:
  /*
   * Constructor uses points from our file to construct our hull
   */
  GrahamScanSerial(const char* filename) : filename_(filename) { ReadFile(); };

  /*
   * Constructor to generate n points on the fly
   *
   * n is the number of points to generate
   */
  GrahamScanSerial(int n) {
    // fixed seed so data doesn't have to be stored
    std::default_random_engine generator(0);
    std::normal_distribution<double> distribution(0.0, 1.0);

    points_.resize(n);
    int idx = 0;
    Point<Num_Type> curr_min;
    for (int i = 0; i < n; i++) {
      Point<Num_Type> current_point;
      current_point.x_ = distribution(generator);
      current_point.y_ = distribution(generator);

      if (i == 0) {
        curr_min = current_point;
      }

      // update the current minumim point's index and value
      if ((current_point.y_ == curr_min.y_ && current_point.x_ < curr_min.x_) ||
          current_point.y_ < curr_min.y_ || idx == 0) {
        curr_min = current_point;
      }

      points_[i] = current_point;
    }
    p0_ = curr_min;
  }

  ~GrahamScanSerial() {}

  /*
   * filename of points to be read in
   */
  std::string filename_;

  /*
   * all of our points from the file
   */
  std::vector<Point<Num_Type> > points_;

  /*
   * our convex hull
   */
  std::vector<Point<Num_Type> > hull_;  // TODO what is this before gethull?

  /*
   * Gets our convex hull using the graham-scan algorithm.  The hull is stored
   * in the public hull_ variable.
   *
   * Serial Implementation
   *
   */
  void GetHullSerial() {
    CenterP0();

    // sort after the first point (p0)
    std::sort(points_.begin() + 1, points_.end());

    // count total number of relevant points in points_
    int total_rel = 1;
    int curr = 1;
    int runner = 2;
    while (runner < points_.size() + 1) {
      // we only want to keep the furthest elt from p0 where multiple points
      // have the same polar angle
      if (runner == points_.size() ||
          GetTurnDir(points_[curr], points_[runner]) != NONE) {
        // if points are now not colinear, take the last colinear point and
        // store it at the last relevant index of points_
        points_[total_rel] = points_[runner - 1];
        curr = runner;
        total_rel++;
      }
      runner++;
    }

    std::stack<Point<Num_Type> > s;
    s.push(points_[0]);
    s.push(points_[1]);
    s.push(points_[2]);

    Point<Num_Type> top, next_to_top, current_point;
    for (int i = 3; i < total_rel; i++) {
      top = s.top();
      s.pop();
      next_to_top = s.top();
      current_point = points_[i];

      // while our current point is not to the left of top, relative to
      // next_to_top
      while (GetTurnDir(current_point - next_to_top, top - next_to_top) !=
             LEFT) {
        top = next_to_top;
        s.pop();
        next_to_top = s.top();
      }
      s.push(top);
      s.push(current_point);
    }

    while (!s.empty()) {
      hull_.push_back(s.top() + p0_);
      s.pop();
    }
  }

  /*
   * Gets our convex hull using the graham-scan algorithm.  The hull is stored
   * in the public hull_ variable.
   *
   * Parallel Implementation
   *
   */
  void GetHullParallel(int num_threads) {
    CenterP0Parallel_MC(num_threads);

    // sort after the first point (p0)
    std::sort(points_.begin() + 1, points_.end());

    // count total number of relevant points in points_
    int total_rel = 1;
    int curr = 1;
    int runner = 2;
    while (runner < points_.size() + 1) {
      // we only want to keep the furthest elt from p0 where multiple points
      // have the same polar angle
      if (runner == points_.size() ||
          GetTurnDir(points_[curr], points_[runner]) != NONE) {
        // if points are now not colinear, take the last colinear point and
        // store it at the last relevant index of points_
        points_[total_rel] = points_[runner - 1];
        curr = runner;
        total_rel++;
      }
      runner++;
    }

    std::stack<Point<Num_Type> > s;
    s.push(points_[0]);
    s.push(points_[1]);
    s.push(points_[2]);

    Point<Num_Type> top, next_to_top, current_point;
    for (int i = 3; i < total_rel; i++) {
      top = s.top();
      s.pop();
      next_to_top = s.top();
      current_point = points_[i];

      // while our current point is not to the left of top, relative to
      // next_to_top
      while (GetTurnDir(current_point - next_to_top, top - next_to_top) !=
             LEFT) {
        top = next_to_top;
        s.pop();
        next_to_top = s.top();
      }
      s.push(top);
      s.push(current_point);
    }

    while (!s.empty()) {
      hull_.push_back(s.top() + p0_);
      s.pop();
    }
  }

 private:
  Point<Num_Type> p0_;

  GrahamScanSerial(void);

  /*
   * reads all points in from filename_ and stores them in our points_ vecor.
   * as we read through the file, we also keep track of the leftmost/min-y
   * point in our file, which we save as p0_
   */
  void ReadFile() {
    // only function that throws is stoi/stod so need try/catch
    try {
      std::string curr_line;
      std::ifstream infile;
      int idx = 0;

      infile.open(filename_);
      if (errno != 0) {
        GPU_GS_PRINT_ERR_LOC();
        perror("infile.open");
        exit(EXIT_FAILURE);
      }

      // get number of points from first line of file
      getline(infile, curr_line);
      if (infile.fail()) {
        GPU_GS_PRINT_ERR_LOC();
        perror("getline");
        exit(EXIT_FAILURE);
      }

      int total_points = stoi(curr_line);

      if (total_points < 4) {
        GPU_GS_PRINT_ERR("Less than four points in input file");
        exit(EXIT_FAILURE);
      }

      points_.resize(total_points);

      // process each line individually of the file
      Point<Num_Type> curr_min;
      int min_y_idx = 0;
      while (!infile.eof()) {
        getline(infile, curr_line);
        if (infile.fail()) {
          GPU_GS_PRINT_ERR_LOC();
          perror("getline");
          exit(EXIT_FAILURE);
        }
        int comma_idx = curr_line.find(',');
        std::string first_num = curr_line.substr(0, comma_idx);
        std::string second_num =
            curr_line.substr(comma_idx + 1, curr_line.length());

        Point<Num_Type> current_point;
        current_point.x_ = static_cast<Num_Type>(stod(first_num));
        current_point.y_ = static_cast<Num_Type>(stod(second_num));

        // update the current minumim point's index and value
        if ((current_point.y_ == curr_min.y_ &&
             current_point.x_ < curr_min.x_) ||
            current_point.y_ < curr_min.y_ || idx == 0) {
          curr_min = current_point;
          min_y_idx = idx;
        }

        points_[idx] = current_point;
        idx++;
      }

      p0_ = curr_min;

      // make sure that the current min is the first element of the array
      Point<Num_Type> tmp = points_[0];
      points_[0] = points_[min_y_idx];
      points_[min_y_idx] = tmp;

      if (total_points != idx) {
        GPU_GS_PRINT_ERR("Incorrect number of points specified by file");
        exit(EXIT_FAILURE);
      }

      infile.close();
      if (infile.fail()) {
        GPU_GS_PRINT_ERR_LOC();
        perror("infile.close");
        exit(EXIT_FAILURE);
      }
    } catch (const std::out_of_range& oor) {
      GPU_GS_PRINT_ERR(oor.what());
      exit(EXIT_FAILURE);
    } catch (const std::invalid_argument& ia) {
      GPU_GS_PRINT_ERR(ia.what());
      exit(EXIT_FAILURE);
    }
  }

  /*
   * centers points_ around p0_ so that p0_ is now the origin
   */
  void CenterP0() {
    for (int i = 0; i < points_.size(); i++) {
      points_[i] = points_[i] - p0_;
    }
  }

  /*
   * centers points_ around p0_ so that p0_ is now the origin
   * a single task for parallel implementation
   */
  void CenterP0Task_MC(int id, int num_threads) {
    for (int i = id; i < points_.size(); i += num_threads) {
      points_[i] = points_[i] - p0_;
    }
  }

  void CenterP0Parallel_MC(int num_threads) {
    int nThreadsToMake = num_threads - 1;
    std::thread workers[num_threads - 1];

    for (int i = 0; i < nThreadsToMake; i++) {
      workers[i] =
          std::thread(&GrahamScanSerial::CenterP0Task_MC, this, i, num_threads);
    }

    CenterP0Task_MC(nThreadsToMake, num_threads);

    for (int i = 0; i < nThreadsToMake; i++) {
      workers[i].join();
    }
  }
};

}  // namespace gpu_graham_scan

#endif  // _GPU_Graham_Scan_
