project(asyflexa)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "icc")

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

set(CMAKE_CXX_FLAGS "-openmp -mkl=sequential")

add_executable(asyflexa main.cpp)