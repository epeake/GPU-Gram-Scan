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
#include <immintrin.h>
#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <iostream>  // get rid of
#include <random>
#include <stack>
#include <string>
#include <type_traits>
#include <vector>

#include "CycleTimer.h"

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

template <class Num_Type>
void BitonicSortPoints(Point<Num_Type>* points, size_t n_points);

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
 *  p: our point
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
  /* Constructor reads all points in from filename_ and stores them in our
   * points_ array. as we read through the file, we also keep track of the
   * leftmost/min-y point in our file, which we save as p0_
   */
  GrahamScanSerial(const char* filename) : filename_(filename) {
    // only function that throws is stoi/stod so need try/catch
    try {
      std::string curr_line;
      std::ifstream infile;
      size_t idx = 0;

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

      n_ = stoi(curr_line);

      if (n_ < 4) {
        GPU_GS_PRINT_ERR("Less than four points in input file");
        exit(EXIT_FAILURE);
      }

      points_ = new Point<Num_Type>[n_];

      // process each line individually of the file
      Point<Num_Type> curr_min;
      size_t min_y_idx = 0;
      while (!infile.eof()) {
        getline(infile, curr_line);
        if (infile.fail()) {
          GPU_GS_PRINT_ERR_LOC();
          perror("getline");
          exit(EXIT_FAILURE);
        }
        size_t comma_idx = curr_line.find(',');
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

      if (n_ != idx) {
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
  };

  /*
   * Constructor to randomly generate n points
   *
   * n is the number of points to generate
   */
  GrahamScanSerial(size_t n) : n_(n) {
    if (n < 4) {
      GPU_GS_PRINT_ERR("Less than four points chosen");
      exit(EXIT_FAILURE);
    }

    // fixed seed so data doesn't have to be stored
    std::default_random_engine generator(0);
    std::uniform_real_distribution<double> distribution(1.0, 10000.0);

    // Aligned allocation (https://stackoverflow.com/a/32612833) where C++17's
    // approach with "new" is not working. Here we use an allocater provided
    // by the intrinsics library (which unfortunately has its own free...).
    points_ = (Point<Num_Type>*)_mm_malloc(n_ * sizeof(Point<Num_Type>), 32);

    Point<Num_Type> curr_min;
    size_t min_y_idx = 0;
    for (size_t i = 0; i < n; i++) {
      Point<Num_Type> current_point;
      current_point.x_ = static_cast<Num_Type>(distribution(generator));
      current_point.y_ = static_cast<Num_Type>(distribution(generator));

      if (i == 0) {
        curr_min = current_point;
      }

      // update the current minumim point's index and value
      if ((current_point.y_ == curr_min.y_ && current_point.x_ < curr_min.x_) ||
          current_point.y_ < curr_min.y_ || i == 0) {
        curr_min = current_point;
        min_y_idx = i;
      }

      points_[i] = current_point;
    }
    p0_ = curr_min;

    // make sure that the current min is the first element of the array
    Point<Num_Type> tmp = points_[0];
    points_[0] = points_[min_y_idx];
    points_[min_y_idx] = tmp;
  }

  ~GrahamScanSerial() { _mm_free(points_); }

  /*
   * filename of points to be read in
   */
  std::string filename_;

  /*
   * all of our points from the file
   */
  Point<Num_Type>* points_;

  /*
   * length of our points
   */
  size_t n_;

  /*
   * our convex hull
   */
  std::vector<Point<Num_Type>> hull_;

  /*
   * Gets our convex hull using the graham-scan algorithm.  The hull is stored
   * in the public hull_ variable.
   *
   * Serial Implementation
   *
   */
  void GetHullSerial() {
    double start_time = CycleTimer::currentSeconds();
    CenterP0();
    double end_time = CycleTimer::currentSeconds();
    std::cout << "center ser: " << (end_time - start_time) * 1000 << " ms\n";

    // sort after the first point (p0)
    std::sort(points_ + 1, points_ + n_);

    // count total number of relevant points in points_
    size_t total_rel = 1;
    size_t curr = 1;
    size_t runner = 2;
    while (runner < n_ + 1) {
      // we only want to keep the furthest elt from p0 where multiple points
      // have the same polar angle
      if (runner == n_ || GetTurnDir(points_[curr], points_[runner]) != NONE) {
        // if points are now not colinear, take the last colinear point and
        // store it at the last relevant index of points_
        points_[total_rel] = points_[runner - 1];
        curr = runner;
        total_rel++;
      }
      runner++;
    }

    std::stack<Point<Num_Type>> s;
    s.push(points_[0]);
    s.push(points_[1]);
    s.push(points_[2]);

    Point<Num_Type> top, next_to_top, current_point;
    for (size_t i = 3; i < total_rel; i++) {
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
  void GetHullParallel() {
    double start_time = CycleTimer::currentSeconds();
    CenterP0Parallel();
    double end_time = CycleTimer::currentSeconds();
    std::cout << "center: " << (end_time - start_time) * 1000 << " ms\n";

    start_time = CycleTimer::currentSeconds();
    // sort after the first point (p0)
    gpu_graham_scan::BitonicSortPoints(points_ + 1, n_ - 1);
    end_time = CycleTimer::currentSeconds();
    std::cout << "sort: " << (end_time - start_time) * 1000 << " ms\n";

    // count total number of relevant points in points_
    size_t total_rel = 1;
    size_t curr = 1;
    size_t runner = 2;
    while (runner < n_ + 1) {
      // we only want to keep the furthest elt from p0 where multiple points
      // have the same polar angle
      if (runner == n_ || GetTurnDir(points_[curr], points_[runner]) != NONE) {
        // if points are now not colinear, take the last colinear point and
        // store it at the last relevant index of points_
        points_[total_rel] = points_[runner - 1];
        curr = runner;
        total_rel++;
      }
      runner++;
    }

    start_time = CycleTimer::currentSeconds();
    std::stack<Point<Num_Type>> s;
    s.push(points_[0]);
    s.push(points_[1]);
    s.push(points_[2]);

    Point<Num_Type> top, next_to_top, current_point;
    for (size_t i = 3; i < total_rel; i++) {
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
    end_time = CycleTimer::currentSeconds();
    std::cout << "hull: " << (end_time - start_time) * 1000 << " ms\n";
  }

 private:
  Point<Num_Type> p0_;

  GrahamScanSerial(void);

  /*
   * centers points_ around p0_ so that p0_ is now the origin
   */
  void CenterP0() {
    for (size_t i = 0; i < n_; i++) {
      points_[i] = points_[i] - p0_;
    }
  }

  /*
   * centers points_ around p0_ so that p0_ is now the origin
   */
  void CenterP0Parallel() {
    if (std::is_same<Num_Type, int32_t>::value) {
      // x values are on the even indicies

      // TODO: error check _mm_malloc
      Point<Num_Type>* pre_x_mask =
          (Point<Num_Type>*)_mm_malloc(4 * sizeof(Point<Num_Type>), 32);
      for (int i = 0; i < 4; i++) {
        pre_x_mask[i] = Point<Num_Type>(0xffffffff, 0);
      }

      __m256i load_mask = _mm256_set1_epi32(0xffffffff);
      __m256i x_mask = _mm256_maskload_epi32((int32_t*)pre_x_mask, load_mask);

      // assemble our p0 vector
      __m256i p0_x_v = _mm256_set1_epi32(p0_.x_);
      __m256i p0_y_v = _mm256_set1_epi32(p0_.y_);
      __m256i p0_v = _mm256_or_si256(_mm256_and_si256(x_mask, p0_x_v),
                                     _mm256_andnot_si256(x_mask, p0_y_v));

      int32_t* p = (int32_t*)points_;

#pragma omp for nowait
      for (size_t i = 0; i < n_ * 2; i += 8) {
        __m256i curr_vals = _mm256_maskload_epi32(p + i, load_mask);
        _mm256_maskstore_epi32(p + i, load_mask,
                               _mm256_sub_epi32(curr_vals, p0_v));
      }
    }

    // if (std::is_same<Num_Type, float>::value) {
    // __m256 x = _mm256_set1_ps(p0_.x_);
    // __m256 y = _mm256_set1_ps(p0_.y_);
    // std::cout << "I am a float";
    // }
    // #pragma omp for nowait
    //     for (size_t i = 0; i < n_; i++) {
    //       points_[i] = points_[i] - p0_;
    //     }
  }
};

}  // namespace gpu_graham_scan

#endif  // _GPU_Graham_Scan_
