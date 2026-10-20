# Hardware connectivity discovery

This document records read-only hardware discovery and offline implementation artifacts. No FPGA
reset, programming, bitstream load, PCIe rescan, BAR write, DMA transfer, driver operation or
benchmark was performed.

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

### YPCB-00338 m4 implementation image

A complete synthesis and implementation was run on 2026-10-17 with Vivado 2024.2.2 (64-bit,
SW Build 6060944), targeting `xc7k480tffg1156-2`, m4 (D=128, MCOLS=4, ACT_ROWS=16, 64K-word
TMEM, MXU_E=4, TMEM_IMPL=3, DDR3-1066, PCIe Gen2 x8, core clock 133.33 MHz). The board
documentation recommends Vivado 2026.1; these results are specific to 2024.2.2.

| Artifact/result | Verified value |
|---|---|
| Bitstream | `build/vivado-m4-20261017-r2/project/otpu.bit`, 13,769,562 bytes |
| Bitstream SHA-256 | `7CE56D7EAE5F97648D3E3E5183C9CEC921F6666D2D569C9F4268F243C580E572` |
| BPI x16 MCS | `build/vivado-m4-20261017-r2/project/otpu.mcs`, 38,730,134 bytes |
| MCS SHA-256 | `419193D809C1D4E0704ABFD825FABBE454F14F8C781C189B138DD65A880E012C` |
| Routed timing | WNS +0.004 ns, WHS +0.012 ns; no failing setup/hold endpoints |
| DRC | No error-severity violations; warnings/advisories remain; no `DSP_PCOUT_may_only_drive_PCIN` |
| Methodology | 12 TIMING-54 Critical Warnings: scoped `Max Delay Datapath Only` constraints between clocks. The existing CDC Tcl documents why these crossings are bounded rather than cut; review the warning before loading the image. |
| Utilization | 194,733 LUTs (65.22%), 170,912 FFs (28.62%), 295 BRAM tiles (30.89%), 734 DSP48E1 (38.23%) |

The first implementation attempt failed placement with 100 `DSP_PCOUT_may_only_drive_PCIN` errors.
The DSP PE was changed so raw `PCOUT` only feeds the DSP cascade, while ternary accumulation uses
a separate registered-sum input. The rebuilt image routed, passed the final DRC report without
errors, met its declared timing constraints by a very small margin, and was written successfully.
The methodology report also has 12 TIMING-54 Critical Warnings for the intentional scoped
CDC max-delay constraints in `constraints/otpu_top_native.tcl` and `constraints/otpu_mem_ch.tcl`.
The constraints keep synchronizer/data crossings timed rather than masking them with broad false
paths; Vivado warns that scoped inter-clock constraints may affect paths beyond their intended
scope. The warnings have not been suppressed or waived and should be reviewed before programming.
The outputs are build artifacts, not source-controlled files.

Neither image has been programmed or flashed; no FPGA reset was issued. This verifies the build
against the specified part in Vivado, not the loaded image identity or operation on the physical
board. The earlier JTAG IDCODE discrepancy remains unresolved. The `.dide\hdl\systemverilog.bin`
file is not a verified programming image and was left untouched.

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
| Vivado m4 synthesis, routed implementation and image generation | **VERIFIED** (2026-10-17; Vivado 2024.2.2; hashes above; not programmed) |
| Linux host/network reachability (SSH TCP port) | **PORT REACHABLE** |
| Trusted SSH server identity | **VERIFIED: user-confirmed ED25519 fingerprint** |
| SSH login | **BLOCKED: configured public-key authentication denied; password prompt unavailable** |
| PCIe endpoint enumeration and driver binding | **NOT RUN** |
| XDMA device readiness / DMA functionality | **NOT RUN** |
| openTPU runtime execution / AI inference | **NOT RUN** |

## 2026-10-20: power loss and recovery

The board lost power (accidental). Recovery, run by the user from `C:\myOpenTPU\_claude`:
1. `recover_1_jtag.cmd` (Vivado 2024.2 batch, `recover_r3.tcl`): JTAG target
   `localhost:3121/xilinx_tcf/Digilent/210299A88B38`, one device xc7k480t_0, the r3 `otpu.bit`
   loaded (volatile), end of startup HIGH, DONE = 1.
2. `recover_2_pcie.cmd`: `sudo opentpu/host/setup_pcie.sh --rescan` on NVX-LPS, then the checks
   (`recover_check.sh`): card 0000:04:00.0 10ee:7028, link 5.0 GT/s **x4** (max x8),
   `/dev/xdma0_*` present, firmware contract build 0xe9727303 D=128 MCOLS=4 LANES=8,
   `otpu-selftest` **ALL PASS** (link, config, calib, scrub 4 GiB, regs, addr, pattern,
   bandwidth host->card 1.67 GB/s / card->host 1.49 GB/s, kernel, vops, stream, tmm, ecc 0/0).
