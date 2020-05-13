cmake_minimum_required(VERSION 3.12)
project(gpu_graham_scan CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
cmake_policy(SET CMP0074 NEW)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specific C++ features

# Not all versions of Clang commonly used have this feature of C++17, even if they support C++17 generally
# so we need to check
include(CheckCXXSourceCompiles)
CHECK_CXX_SOURCE_COMPILES(
  "#include <new>
  int main() { float* values = new (std::align_val_t(32)) float[100]; return 0; }"
  HAVE_ALIGN_VAL
)
if (HAVE_ALIGN_VAL)
  add_compile_definitions(HAVE_ALIGN_VAL)
endif()

# set(CMAKE_CXX_FLAGS "-m64 -Wall -march=native -Wno-unknown-pragmas -Wno-unused-function  -O3 -g")
set(CMAKE_CXX_FLAGS "-m64 -Wall -march=native -Wno-unknown-pragmas -Wno-unused-function  -O3")


OPTION(DEFINE_VERBOSE
  "Build the project using verbose code"
  OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# CUDA Support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "Could not find the CUDA compiler and toolkit. If you have CUDA-compatible GPU install the CUDA compiler/toolkit.")
endif()

# OpenMP Support
if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  find_package(OSXOpenMP)
else()
  find_package(OpenMP)
endif()
if(NOT OpenMP_FOUND)
  message(STATUS "Could not find compiler with OpenMP support.")
endif()

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(headerFiles 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bitonic_sort.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_graham_scan.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_graham_scan_test.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda-util.h
)

set_source_files_properties(
    tasksys.cc
    PROPERTIES
    COMPILE_FLAGS -Wno-unused-function
)

find_package(Boost
  COMPONENTS unit_test_framework
  REQUIRED
)

add_executable(project-main
  test_main.cc
  ${headerFiles}
)
if (OpenMP_FOUND)
  target_link_libraries(project-main PUBLIC OpenMP::OpenMP_CXX)
endif()


# unit testing
# adapted from https://www.neyasystems.com/2014/06/20/an-engineers-guide-to-unit-testing-cmake-and-boost-unit-tests/
# and https://eb2.co/blog/2015/06/driving-boost.test-with-cmake/

# Setup CMake to run tests
enable_testing()

# Prep ourselves for compiling boost
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})
message(STATUS "Found Boost: ${Boost_VERSION}")

# Keep test files in a separate source directory called test
file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/*.cc)

# Run through each source
foreach(testSrc ${TEST_SRCS})

  # Extract the filename without an extension (NAME_WE)
  get_filename_component(testName ${testSrc} NAME_WE)

  # Add compile target
  add_executable(${testName} ${testSrc} ${headerFiles})

  # link to Boost libraries AND your targets and dependencies
  target_link_libraries(${testName} ${Boost_LIBRARIES})
  set_target_properties(${testName} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testBin)

  file(READ "${testSrc}" testSrcContents)
    string(REGEX MATCHALL "BOOST_AUTO_TEST_CASE\\( *([A-Za-z_0-9]+) *\\)" 
           foundTests ${testSrcContents})

    foreach(HIT ${foundTests})
        message(${HIT})
        string(REGEX REPLACE ".*\\( *([A-Za-z_0-9]+) *\\).*" "\\1" subTestName ${HIT})
    
        add_test(NAME "${testName}.${subTestName}" 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testBin
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testBin/${testName}
                 --log_level=message
                 --run_test=${subTestName} --catch_system_error=yes)
    endforeach()

endforeach(testSrc)
