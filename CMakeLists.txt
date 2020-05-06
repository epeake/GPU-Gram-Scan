cmake_minimum_required(VERSION 3.12)
project(cs416_prog CXX)

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

set(CMAKE_CXX_FLAGS "-m64 -Wall -O3")

OPTION(DEFINE_VERBOSE
  "Build the project using verbose code"
  OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ISPC Support
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddISPCObject.cmake)
find_program(ISPC_EXECUTABLE ispc DOC "Path to the ISPC executable.")
if (NOT ISPC_EXECUTABLE)
    message(STATUS "Could not find Intel SPMD Compiler (ispc). Make sure you have completed the 'Getting Started' instructions on Canvas.")
else()
    message(STATUS "Found Intel SPMD Compiler (ispc): ${ISPC_EXECUTABLE}")
endif()
set(ISPC_ARCH "x86-64")
# Specify arguments (as a list)
set(ISPC_FLAGS "-O3;--math-lib=default")

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
  message(STATUS "Could not find compiler with OpenMP support. Make sure you have completed the 'Getting Started' instructions on Canvas.")
endif()

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_source_files_properties(
    tasksys.cc
    PROPERTIES
    COMPILE_FLAGS -Wno-unused-function
)

add_library(common_objs
    OBJECT
    tasksys.cc
)
target_link_libraries(common_objs PUBLIC Threads::Threads)



find_package(CBLAS)
if (NOT CBLAS_FOUND)
    message(FATAL_ERROR "Could not find CBLAS library. Exiting.")
endif()
include_directories(${CBLAS_INCLUDE_DIRS})

find_package(CUB)
if (NOT CUB_FOUND)
  message(FATAL_ERROR, "Could not CUB library. Exiting.")
endif()
include_directories(${CUB_INCLUDE_DIRS})



# Add current directory to include search path (so tests can find headers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(Boost
  REQUIRED
)
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS "Found Boost: ${Boost_VERSION}")

# Modify this file as you need for your project (modeled on the most relevant assignment), e.g.
# uncomment the following and add the relevant files for your project
add_executable(project-main
  test_main.cc
  gpu_graham_scan.cc
)
target_link_libraries(project-main)