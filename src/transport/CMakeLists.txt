add_library(micro_ccl_transport STATIC
  transport.cc
)

target_include_directories(micro_ccl_transport PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(micro_ccl_transport PUBLIC micro_ccl_verbs)
