cmake_minimum_required(VERSION 3.20)
project(transformer_go_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native
  cpp/native.cpp
  cpp/vendor/katago/core/bsearch.cpp
  cpp/vendor/katago/core/global.cpp
  cpp/vendor/katago/core/hash.cpp
  cpp/vendor/katago/core/md5.cpp
  cpp/vendor/katago/core/rand.cpp
  cpp/vendor/katago/core/rand_helpers.cpp
  cpp/vendor/katago/core/sha2.cpp
  cpp/vendor/katago/core/timer.cpp
  cpp/vendor/katago/game/board.cpp
  cpp/vendor/katago/game/boardhistory.cpp
  cpp/vendor/katago/game/rules.cpp
)
target_compile_options(_native PRIVATE -O3 -Wall -Wextra)
set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/transformer_go")
