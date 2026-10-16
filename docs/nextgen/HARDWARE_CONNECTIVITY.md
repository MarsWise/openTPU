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

### Repeat read-only discovery

A second read-only Vivado Hardware Manager query was run on 2026-10-16 using the saved
`jtag_readonly.tcl` script:

| Item | Observation |
|---|---|
| Vivado | 2024.2.2 (64-bit), SW Build 6060944 |
| JTAG target | `localhost:3121/xilinx_tcf/Digilent/210299A88B38` |
| Device | `xc7k480t_0`, part property `xc7k480t` |
| IDCODE returned by Vivado | `00100011011101010001000010010011` |
| Operations | Enumerated target/device properties; no programming or reset |

The new IDCODE string differs in length and value from the earlier recorded string above.
Treat the current value as the exact fresh Vivado output; the discrepancy is unresolved and is
not evidence of a loaded bitstream identity or a board change. `hw_server` was started for this
query and stopped afterward.

No `.bit` or `.mcs` programming image was found in the feature worktree or the standard
`C:\Users\user\otpu-build` location. The `.dide\hdl\systemverilog.bin` file is not a verified
FPGA programming image and was left untouched. The user authorized programming/reset only
after verifying the exact image; no image was available to verify, so neither operation was
attempted.

## Linux PCIe host

Target supplied by the user: `nvx@192.168.0.156`.

- TCP port 22 is reachable from the Windows host.
- SSH host-key lookup had no matching existing `known_hosts` entry.
- Read-only key scan returned the following keys; the user confirmed the ED25519 fingerprint:
  - ED25519: `SHA256:gtUqtw9OJFkgCfuX0iNNJH3RAMzGmiC7KGOE4x9EDg0`
  - RSA: `SHA256:YMJ6xSOEbEfk7hERz9xqJHj5kjQUjbNXaFov2FBcUVQ`
  - ECDSA: `SHA256:Ms/pqdh+q+5TIVRxPXPXB2368zlHpQv00zCp/JX5x84`
- The confirmed ED25519 key was added to a session-private `known_hosts` file and verified with
  `ssh-keygen`. A key-only, `BatchMode=yes` connection attempt returned
  `Permission denied (publickey,password)`.
- A retry using the currently configured Windows ED25519 identity was also rejected; no remote
  command ran. Its public-key fingerprint is `SHA256:Nqth0L6ArgQ5zX0hW5hKgEKWoAz+84wGP5wzPcoKkMg`.
- No password was sent and no remote command ran. The current command tool cannot provide a
  secure interactive password prompt; use key-based SSH authentication or have the user run the
  read-only diagnostics and share their output.
- PCIe endpoint/BDF, vendor/device ID, driver/module, negotiated link, BARs, AER, XDMA nodes and
  Linux runtime readiness: **NOT INSPECTED**.

## Bring-up evidence ladder

| Milestone | Status |
|---|---|
| Windows USB-JTAG target/device enumeration | **VERIFIED READ-ONLY** (repeated 2026-10-16; IDCODE discrepancy noted) |
| Linux host/network reachability (SSH TCP port) | **PORT REACHABLE** |
| Trusted SSH server identity | **VERIFIED: user-confirmed ED25519 fingerprint** |
| SSH login | **BLOCKED: configured public-key authentication denied; password prompt unavailable** |
| PCIe endpoint enumeration and driver binding | **NOT RUN** |
| XDMA device readiness / DMA functionality | **NOT RUN** |
| openTPU runtime execution / AI inference | **NOT RUN** |
