# Setup: Soft-RoCE on two Ubuntu 24.04 VMs

micro-ccl needs no InfiniBand hardware for development. It runs against
Soft-RoCE (`rdma_rxe`), the Linux kernel's software RoCE provider, on
ordinary Ethernet. The code makes no Soft-RoCE-specific assumptions, so it
should run unmodified on real InfiniBand or RoCE hardware.

## Prerequisites

Two Ubuntu 24.04 VMs (or bare-metal/cloud instances) on the same L2 network,
each able to reach the other's IP.

## Install packages (run on both VMs)

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libibverbs-dev librdmacm-dev rdma-core ibverbs-utils \
    perftest net-tools catch2

# Optional: only needed for the benchmark harness's OpenMPI comparison
# mode (bench_mpi_allreduce). The core library and its own benchmark
# binary (bench_allreduce) build fine without this.
sudo apt install -y libopenmpi-dev openmpi-bin
```

## Bring up rdma_rxe (run on both VMs)

```bash
sudo modprobe rdma_rxe

# Identify your primary NIC first.
ip -br link show

# Bind rxe to that NIC. Replace eth0 with your actual interface name.
sudo rdma link add rxe0 type rxe netdev eth0
```

## Verify

```bash
rdma link show
ibv_devinfo
```

`ibv_devinfo` should show a device (typically `rxe0`) with `state:
PORT_ACTIVE` and `link_layer: Ethernet`. Note the GID at index 0 or 1 --
Soft-RoCE has no LIDs, only GIDs (RoCE v2 encapsulates verbs traffic over
UDP/IP), and the GID is what stage 2's bootstrap exchange will need.

## Cross-node link test with rping

On VM-A (server), substituting its real IP:

```bash
rping -s -a 10.0.0.1 -v
```

On VM-B (client):

```bash
rping -c -a 10.0.0.1 -v -C 10
```

10 completed iterations with matching pattern data in both directions means
the RDMA path between the two VMs works end to end -- QP setup, memory
registration, and RDMA send/recv are all functioning. That's the signal to
move on to stage 2 (pingpong).

## Persisting the rxe link across reboot (optional)

Dev VMs don't strictly need this, but if you want it to survive a reboot,
add the `rdma link add` command to a systemd unit or a netplan hook.
