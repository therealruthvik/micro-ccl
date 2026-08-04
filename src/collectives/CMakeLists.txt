add_library(micro_ccl_collectives STATIC
  reduce_ops.cc
  chunking.cc
  broadcast.cc
  allgather.cc
  allreduce_ring.cc
  allreduce_recursive_doubling.cc
)

target_include_directories(micro_ccl_collectives PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(micro_ccl_collectives PUBLIC
  micro_ccl_verbs
  micro_ccl_communicator
  micro_ccl_transport
)
