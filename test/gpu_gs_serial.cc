// Link to Boost
#define BOOST_TEST_DYN_LINK

// Define our Module name (prints at testing)
#define BOOST_TEST_MODULE "gpu_gs_serial"

#include <vector>

#include "gpu_graham_scan.h"

// VERY IMPORTANT - include this last
#include <boost/test/unit_test.hpp>

// ------------- Tests Follow --------------
int kPoints = 10000;

BOOST_AUTO_TEST_CASE(seeding_works) {
  gpu_graham_scan::GrahamScanSerial<int> random1(kPoints);
  gpu_graham_scan::GrahamScanSerial<int> random2(kPoints);

  bool is_same = true;
  if (random1.n_ != random2.n_) {
    is_same = false;
  } else {
    for (size_t i = 0; i < random1.n_; i++) {
      if (random1.points_[i] != random2.points_[i]) {
        is_same = false;
      }
    }
  }

  BOOST_CHECK(is_same);
}
