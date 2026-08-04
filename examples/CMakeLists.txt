add_executable(pingpong pingpong.cc)
target_link_libraries(pingpong PRIVATE
  micro_ccl_verbs
  micro_ccl_bootstrap
  micro_ccl_transport
)
