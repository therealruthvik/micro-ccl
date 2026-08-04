add_library(micro_ccl_verbs STATIC
  device.cc
  protection_domain.cc
  memory_region.cc
  completion_queue.cc
  queue_pair.cc
  endpoint_info.cc
)

target_include_directories(micro_ccl_verbs PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(micro_ccl_verbs PUBLIC ibverbs::ibverbs)
