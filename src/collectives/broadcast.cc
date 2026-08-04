#include "micro_ccl/collectives/broadcast.hpp"

#include <stdexcept>
#include <string>

#include "micro_ccl/transport.hpp"

namespace micro_ccl {

void broadcast(Communicator& comm, MemoryRegion& mr, void* buf, size_t count,
                Dtype dtype, int root) {
  if (root < 0 || root >= comm.size()) {
    throw std::invalid_argument("broadcast: root out of range");
  }
  size_t bytes = count * dtype_size(dtype);

  if (comm.rank() == root) {
    for (int peer = 0; peer < comm.size(); ++peer) {
      if (peer == root) continue;
      transport::send(comm.qp(peer), comm.send_cq(), mr, buf, bytes);
    }
  } else {
    size_t got =
        transport::recv(comm.qp(root), comm.recv_cq(), mr, buf, bytes);
    if (got != bytes) {
      throw std::runtime_error(
          "broadcast: size mismatch, expected " + std::to_string(bytes) +
          " bytes from root but got " + std::to_string(got) +
          " -- ranks disagree on message size");
    }
  }
}

}  // namespace micro_ccl
