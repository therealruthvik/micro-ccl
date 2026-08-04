add_executable(bench_allreduce bench_allreduce.cc)
target_link_libraries(bench_allreduce PRIVATE
  micro_ccl_verbs
  micro_ccl_bootstrap
  micro_ccl_transport
  micro_ccl_communicator
  micro_ccl_collectives
)

# The OpenMPI comparison binary is optional: the rest of the project must
# build without OpenMPI installed (it is not one of the project's core
# dependencies, only the comparison harness's), so this target only gets
# added when CMake actually finds an MPI C++ implementation.
find_package(MPI COMPONENTS CXX)
if(MPI_FOUND)
  add_executable(bench_mpi_allreduce bench_mpi_allreduce.cc)
  target_link_libraries(bench_mpi_allreduce PRIVATE MPI::MPI_CXX)
else()
  message(STATUS
    "MPI not found -- skipping bench_mpi_allreduce (install libopenmpi-dev "
    "and openmpi-bin to enable the OpenMPI comparison mode)")
endif()
