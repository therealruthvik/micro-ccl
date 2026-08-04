cmake_minimum_required(VERSION 3.16)
project(micro_ccl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)

# ibverbs has no CMake package config on most distros, so locate it manually.
find_path(IBVERBS_INCLUDE_DIR NAMES infiniband/verbs.h)
find_library(IBVERBS_LIBRARY NAMES ibverbs)
if(NOT IBVERBS_INCLUDE_DIR OR NOT IBVERBS_LIBRARY)
  message(FATAL_ERROR
    "libibverbs not found. On Ubuntu: sudo apt install libibverbs-dev. "
    "This project only builds on Linux with verbs headers present.")
endif()

add_library(ibverbs::ibverbs UNKNOWN IMPORTED)
set_target_properties(ibverbs::ibverbs PROPERTIES
  IMPORTED_LOCATION "${IBVERBS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}")

enable_testing()

include_directories(include)

add_subdirectory(src/verbs)
add_subdirectory(src/bootstrap)
add_subdirectory(src/transport)
add_subdirectory(examples)
