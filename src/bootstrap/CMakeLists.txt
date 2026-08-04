add_library(micro_ccl_bootstrap STATIC
  tcp_bootstrap.cc
)

target_include_directories(micro_ccl_bootstrap PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(micro_ccl_bootstrap PUBLIC micro_ccl_verbs)
