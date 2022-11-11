cmake_minimum_required(VERSION 3.4...3.18)
project(lanms)
find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)
pybind11_add_module(adaptor adaptor.cpp clipper/clipper.cpp)

