# CDC sign-off: r5x lean ed482f26 (2026-11-24)

Source: `reports/cdc.rpt` of `build/vivado-r8-m8-133.33-ed482f26-lean` (Vivado 2024.2.2,
`report_cdc -details`, routed). Evidence class: SOURCE INSPECTION of the report and the RTL; the
constraint changes below take effect in the next build (NOT RUN).

## Summary of the report

| ID | Severity | Rows | Waived endpoints | Verdict |
|---|---|---|---|---|
| CDC-3 / CDC-9 | Info | 51 / 2 | - | Proper ASYNC_REG synchronizers; nothing to do |
| CDC-26 | Warning | 96 | 1024 | Distributed-RAM FIFO heads read across clocks (otpu_mem_ch and LiteX FIFOs); safe by the pointer protocol |
| CDC-1 | Critical | 4 | 648 | otpu_mem_ch `g_xt.x_ripe` (XDMA batch timer, added after the waiver list was written): safe, now waived (below) |
| CDC-15 | Warning | 36 | 748 | otpu_mem_ch `g_xt.x_age[*]`: the same, now waived |
| CDC-6 | Warning | 10 | 260 | LiteDRAM's CSR AXI-Lite crossing, 3-bit gray pointers under LiteX's false path: skew now bounded by set_bus_skew 7.5 ns, then waived |
| CDC-10 | Critical | 2 | - | LiteDRAM CRG reset `crg_rst0 = rst \| (reset_wr_stb & reset_storage[0])` into the MMCM / PLL reset registers (50 MHz): the host never writes `ctrl_reset`, so the term is constant; waived with that condition |
| CDC-13 | Critical | 2 | 10440 | XDMA's `pipe_clock` pclk_sel into its BUFGCTRL select (Xilinx IP, by design): waived |

## The cases

- **`g_xt` (otpu_mem_ch, XDMA batches).** `x_age` / `x_ripe` are reset whenever `x_ok` (the
  input FIFO's registered not-empty AND its head's bank bit) is low, or a command is taken. While
  the FIFO is empty its head may be changing, but `x_ok` is then 0 whatever the head shows, so
  the registers load a constant. While it is not empty the head is stable (written two cycles
  before its write pointer crossed, not rewritten before the read pointer crosses back). The
  same argument as every other head consumer in `otpu_mem_ch.tcl`; the waiver list lacked the
  two registers.
- **LiteDRAM's CSR crossing.** LiteX's AXI CDC moves the CSR bus between the core clock and the
  controller clock through 4-entry async FIFOs with 3-bit gray pointers. LiteX cuts the
  synchronizer inputs with `set_false_path` (`mr_ff`), which leaves the bus skew unconstrained; a
  gray pointer is safe only while its bits arrive within one destination period. `set_bus_skew
  7.5` (both clocks are 7.5 ns) restores that bound.
- **LiteDRAM's CRG reset.** `soc_rst` is the ctrl_reset CSR's write strobe ANDed with its bit 0.
  The host does not write that CSR (no use under `opentpu/host`, only `csr.csv`), so bit 0 stays
  0 and the path is constant. If a host change ever writes ctrl_reset, the pulse (one 133 MHz
  cycle) is also shorter than the 50 MHz sampling period: a reset through it would need a
  stretched, registered pulse first.
- **XDMA pclk_sel.** Inside Xilinx's PCIe IP (the Gen1 / Gen2 clock switch); the IP's own
  constraints treat it as a false path.

## Remaining

- Next build: `reports/cdc.rpt` must show no unwaived Critical row, and `reports/bus_skew.rpt`
  the new LiteDRAM constraint met.
- Not covered by report_cdc: paths that cross through ports or unclocked logic (check_timing.rpt
  has none of the "no_clock" / "unconstrained_internal_endpoints" kind in ed482f26).
