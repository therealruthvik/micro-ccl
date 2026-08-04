add_library(micro_ccl_communicator STATIC
  communicator.cc
)

target_include_directories(micro_ccl_communicator PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(micro_ccl_communicator PUBLIC
  micro_ccl_verbs
  micro_ccl_bootstrap
)
