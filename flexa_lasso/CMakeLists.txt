project(flexa)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "mpicxx")

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

set(CMAKE_CXX_FLAGS "-cxx=icpc -mkl=sequential")

add_executable(flexa main.cpp)
