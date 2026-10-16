# Hardware connectivity discovery

This document records read-only discovery only. No FPGA reset, programming, bitstream load,
PCIe rescan, BAR write, DMA transfer, driver operation or benchmark was performed.

## Windows JTAG host

Checked 2026-10-16 using the repository development machine:

| Item | Observation |
|---|---|
| Vivado | 2024.2.2 (64-bit), SW Build 6060944 |
| hw_server | Started temporarily on local TCP 3121 for discovery, then stopped |
| Cable | Digilent, serial `210299A88B38` |
| JTAG target | `localhost:3121/xilinx_tcf/Digilent/210299A88B38` |
| Device | `xc7k480t_0`, part property `xc7k480t` |
| IDCODE | `001000110111010010010011` |
| Operations | Opened Hardware Manager target and enumerated target/device properties only |

This verifies JTAG visibility and the reported FPGA part/IDCODE at the time of discovery. It does
not establish which bitstream is loaded, FPGA application state, PCIe enumeration, driver
readiness, DMA operation, or inference.

## Linux PCIe host

Target supplied by the user: `nvx@192.168.0.156`.

- TCP port 22 is reachable from the Windows host.
- SSH host-key lookup had no matching existing `known_hosts` entry.
- Read-only key scan returned:
  - ED25519: `SHA256:gtUqtw9OJFkgCfuX0iNNJH3RAMzGmiC7KGOE4x9EDg0`
  - RSA: `SHA256:YMJ6xSOEbEfk7hERz9xqJHj5kjQUjbNXaFov2FBcUVQ`
  - ECDSA: `SHA256:Ms/pqdh+q+5TIVRxPXPXB2368zlHpQv00zCp/JX5x84`
- No SSH authentication or remote commands were attempted because the host identity could not
  be matched to a previously trusted key. Do not send the supplied password until the user
  confirms the expected host-key fingerprint or installs the correct host key through a trusted
  channel.
- PCIe endpoint/BDF, vendor/device ID, driver/module, negotiated link, BARs, AER, XDMA nodes and
  Linux runtime readiness: **NOT INSPECTED**.

## Bring-up evidence ladder

| Milestone | Status |
|---|---|
| Windows USB-JTAG target/device enumeration | **VERIFIED READ-ONLY** |
| Linux host/network reachability (SSH TCP port) | **PORT REACHABLE** |
| Trusted SSH server identity and login | **BLOCKED: fingerprint confirmation required** |
| PCIe endpoint enumeration and driver binding | **NOT RUN** |
| XDMA device readiness / DMA functionality | **NOT RUN** |
| openTPU runtime execution / AI inference | **NOT RUN** |
