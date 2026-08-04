#include "micro_ccl/bootstrap.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

namespace micro_ccl {
namespace {

// send()/recv() are permitted by POSIX to transfer fewer bytes than
// requested even on a healthy blocking socket (short writes under
// backpressure, short reads if the peer's data hasn't all arrived yet).
// Every bootstrap message has a known fixed size, so these loop until
// exactly that many bytes have moved, and turn "peer closed early" into an
// explicit error instead of silently handing back a truncated struct that
// would go on to corrupt a QP handshake.
void send_exact(int fd, const void* buf, size_t len) {
  const auto* p = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, p + sent, len - sent, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("bootstrap send failed: ") +
                                std::strerror(errno));
    }
    sent += static_cast<size_t>(n);
  }
}

void recv_exact(int fd, void* buf, size_t len) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, p + got, len - got, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("bootstrap recv failed: ") +
                                std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error(
          "bootstrap peer closed connection mid-message");
    }
    got += static_cast<size_t>(n);
  }
}

void send_u32(int fd, uint32_t v) {
  uint32_t be = htonl(v);
  send_exact(fd, &be, sizeof(be));
}

uint32_t recv_u32(int fd) {
  uint32_t be = 0;
  recv_exact(fd, &be, sizeof(be));
  return ntohl(be);
}

// One (destination rank, endpoint info) pair as it travels on the wire:
// 4 bytes dst_rank + the fixed-size encoded QpEndpointInfo. The source rank
// does not need its own separate field -- QpEndpointInfo already carries
// `rank`, set by whoever produced it.
constexpr size_t kEntryWireSize = 4 + kEndpointInfoWireSize;

void send_entry(int fd, int dst_rank, const QpEndpointInfo& info) {
  std::array<uint8_t, kEntryWireSize> buf{};
  uint32_t be_dst = htonl(static_cast<uint32_t>(dst_rank));
  std::memcpy(buf.data(), &be_dst, 4);
  std::array<uint8_t, kEndpointInfoWireSize> encoded{};
  encode_endpoint_info(info, encoded);
  std::memcpy(buf.data() + 4, encoded.data(), kEndpointInfoWireSize);
  send_exact(fd, buf.data(), buf.size());
}

std::pair<int, QpEndpointInfo> recv_entry(int fd) {
  std::array<uint8_t, kEntryWireSize> buf{};
  recv_exact(fd, buf.data(), buf.size());
  uint32_t be_dst = 0;
  std::memcpy(&be_dst, buf.data(), 4);
  int dst_rank = static_cast<int>(ntohl(be_dst));
  std::array<uint8_t, kEndpointInfoWireSize> encoded{};
  std::memcpy(encoded.data(), buf.data() + 4, kEndpointInfoWireSize);
  return {dst_rank, decode_endpoint_info(encoded)};
}

int connect_with_retry(const std::string& host, uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  int gai = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints,
                         &res);
  if (gai != 0) {
    throw std::runtime_error("getaddrinfo(" + host +
                              ") failed: " + gai_strerror(gai));
  }

  // Rank launch order is not coordinated -- a non-root rank may start
  // before rank 0's listen socket is up. Rather than requiring the caller
  // to orchestrate startup order, retry the connect for a few seconds; this
  // is a one-time setup cost, not data-path code, so simplicity wins over
  // an event-driven approach here.
  const int kMaxAttempts = 50;
  const auto kDelay = std::chrono::milliseconds(200);
  std::string last_error;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      freeaddrinfo(res);
      throw std::runtime_error(std::string("socket() failed: ") +
                                std::strerror(errno));
    }
    if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
      freeaddrinfo(res);
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      return fd;
    }
    last_error = std::strerror(errno);
    ::close(fd);
    std::this_thread::sleep_for(kDelay);
  }
  freeaddrinfo(res);
  throw std::runtime_error("could not connect to " + host + ":" +
                            std::to_string(port) +
                            " after retries, last error: " + last_error);
}

}  // namespace

TcpBootstrap::TcpBootstrap(int rank, int world_size,
                            const std::string& root_host, uint16_t root_port)
    : rank_(rank), world_size_(world_size) {
  if (rank == 0) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
      throw std::runtime_error(std::string("socket() failed: ") +
                                std::strerror(errno));
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(root_port);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
               sizeof(addr)) != 0) {
      throw std::runtime_error(std::string("bind() failed: ") +
                                std::strerror(errno));
    }
    if (::listen(listen_fd_, world_size) != 0) {
      throw std::runtime_error(std::string("listen() failed: ") +
                                std::strerror(errno));
    }

    // peer_fds_[0] (self) is never used over a socket; sized to world_size
    // so peer_fds_[r] can be indexed directly by rank.
    peer_fds_.assign(static_cast<size_t>(world_size), -1);
    for (int i = 0; i < world_size - 1; ++i) {
      int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        throw std::runtime_error(std::string("accept() failed: ") +
                                  std::strerror(errno));
      }
      int one_nodelay = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one_nodelay,
                   sizeof(one_nodelay));
      // Each connecting rank announces its own rank as the very first
      // thing it sends, so the root can file the connection under the
      // right index regardless of connection/accept ordering.
      uint32_t their_rank = recv_u32(fd);
      if (their_rank == 0 || their_rank >= static_cast<uint32_t>(world_size)) {
        ::close(fd);
        throw std::runtime_error("bootstrap: peer announced invalid rank " +
                                  std::to_string(their_rank));
      }
      peer_fds_[their_rank] = fd;
    }
  } else {
    conn_fd_ = connect_with_retry(root_host, root_port);
    send_u32(conn_fd_, static_cast<uint32_t>(rank));
  }
}

TcpBootstrap::~TcpBootstrap() {
  if (conn_fd_ >= 0) ::close(conn_fd_);
  if (listen_fd_ >= 0) ::close(listen_fd_);
  for (int fd : peer_fds_) {
    if (fd >= 0) ::close(fd);
  }
}

std::unordered_map<int, QpEndpointInfo> TcpBootstrap::exchange(
    const std::unordered_map<int, QpEndpointInfo>& outgoing) {
  // Full table of every (dst_rank, info) pair contributed by every rank,
  // assembled on rank 0 and reflected back out. Small-scale by design: this
  // is O(world_size^2) work and memory on rank 0, which is the star-
  // topology tradeoff noted on TcpBootstrap -- fine for the handful of
  // ranks this project targets, not meant to scale to a large cluster.
  std::vector<std::pair<int, QpEndpointInfo>> table;

  if (rank_ == 0) {
    for (const auto& [dst, info] : outgoing) {
      table.emplace_back(dst, info);
    }
    for (int r = 1; r < world_size_; ++r) {
      uint32_t count = recv_u32(peer_fds_[r]);
      for (uint32_t i = 0; i < count; ++i) {
        table.push_back(recv_entry(peer_fds_[r]));
      }
    }

    uint32_t total = static_cast<uint32_t>(table.size());
    for (int r = 1; r < world_size_; ++r) {
      send_u32(peer_fds_[r], total);
      for (const auto& [dst, info] : table) {
        send_entry(peer_fds_[r], dst, info);
      }
    }
  } else {
    send_u32(conn_fd_, static_cast<uint32_t>(outgoing.size()));
    for (const auto& [dst, info] : outgoing) {
      send_entry(conn_fd_, dst, info);
    }

    uint32_t total = recv_u32(conn_fd_);
    table.reserve(total);
    for (uint32_t i = 0; i < total; ++i) {
      table.push_back(recv_entry(conn_fd_));
    }
  }

  std::unordered_map<int, QpEndpointInfo> result;
  for (const auto& [dst, info] : table) {
    if (dst == rank_) {
      result[static_cast<int>(info.rank)] = info;
    }
  }
  return result;
}

}  // namespace micro_ccl
