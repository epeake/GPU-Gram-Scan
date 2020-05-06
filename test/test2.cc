// Link to Boost
#define BOOST_TEST_DYN_LINK

// Define our Module name (prints at testing)
#define BOOST_TEST_MODULE "BaseClassModule"

// VERY IMPORTANT - include this last
#include <boost/test/unit_test.hpp>
// #include "some_project/some_base_class.h"

// ------------- Tests Follow --------------

BOOST_AUTO_TEST_CASE(assignment) {
  int x = 1;
  BOOST_CHECK_EQUAL(x, 1);
}

BOOST_AUTO_TEST_CASE(a) {
  int x = 1;
  BOOST_CHECK_EQUAL(x, 1);
}

BOOST_AUTO_TEST_CASE(sdf) {
  int x = 2;
  BOOST_CHECK_EQUAL(x, 1);
}
