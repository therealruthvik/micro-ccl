find_package(Catch2 3 REQUIRED)
include(Catch)

# Unit tests: pure functions, no verbs/network involved, run anywhere
# (including CI with no RDMA hardware at all).
add_executable(unit_tests
  unit_reduce_ops.cc
  unit_chunking.cc
)
target_link_libraries(unit_tests PRIVATE
  micro_ccl_collectives
  Catch2::Catch2WithMain
)
catch_discover_tests(unit_tests)

# Integration tests: require a real RDMA device (Soft-RoCE or hardware).
# Each is a self-contained multi-process binary (fork()s one process per
# rank) rather than a Catch2 test case, since Catch2's model assumes a
# single process -- see docs/setup.md before running these.
add_executable(integration_allreduce_4rank integration_allreduce_4rank.cc)
target_link_libraries(integration_allreduce_4rank PRIVATE
  micro_ccl_collectives
  micro_ccl_communicator
)
add_test(NAME integration_allreduce_4rank COMMAND integration_allreduce_4rank)

add_executable(integration_mismatched_scratch integration_mismatched_scratch.cc)
target_link_libraries(integration_mismatched_scratch PRIVATE
  micro_ccl_collectives
  micro_ccl_communicator
)
add_test(NAME integration_mismatched_scratch COMMAND integration_mismatched_scratch)
