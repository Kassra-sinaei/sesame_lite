# The TSDF output sink for Hydra, built as one library that is both a Hydra plugin (loaded with
# hydra_python.init_plugins) and a Python module (import sesame_hydra). Built against the Hydra install in
# third_party/install; see outputs/ENV.md for the commands.
cmake_minimum_required(VERSION 3.16)
project(sesame_hydra CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hydra REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sesame_hydra tsdf_sink.cpp)
target_link_libraries(sesame_hydra PRIVATE hydra::hydra)
