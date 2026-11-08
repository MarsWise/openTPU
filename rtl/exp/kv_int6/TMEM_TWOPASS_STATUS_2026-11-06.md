# Experimental TMEM-to-INT6 two-pass row path

## Subsequent eight-bank PBR and pumped-clock update

The reader now emits registered per-bank addresses and read enables for an
eight-bank TMEM. XSim checks the bank address mapping for all eight rotations,
grant stalls and backpressure. The combined path was also tested with real
`otpu_tmem` IMPL=3/RPL=1, PBR enabled and PMP=1 on aligned 1x/2x simulated
clocks, as well as PMP=2. In both cases the selected scenario exercises a
competing port, replay, accepted-request stalls, output writes and reset.

For PMP=1, XSim cannot elaborate the original phase registers with both
`always_ff` and simulation initialization, so the test uses a temporary copy
of TMEM that retains the original phase initialization and replaces only the
two phase-register `always_ff` statements by `always`. Their assignments and
the memory/replay datapaths are unchanged. `SYNTHESIS` still disables TMEM's
other simulation-only checks; replay bookkeeping is explicitly initialized by
the bench. This is a functional RTL simulation, not proof of physical clock
phase, CDC, slice arbitration or timing. PBR is implemented for eight banks;
other bank counts are not supported by this experimental interface.
The selected reader/replay/combined-path/two row-writer series completed
**16 passed** after this update. Earlier simulator failures and scope limits
below still apply; this is not the complete project regression.

Updated OOC for the whole experimental hierarchy, Vivado 2024.2.2,
xc7k480tffg1156-2, 7.500 ns: **3719 LUT, 4475 FF, 18 DSP**, WNS +1.618 ns,
WHS +0.123 ns. No placement, route, full IO constraints or `HD.CLK_SRC`.
Reports: `build/logs/qst6_tmem_twopass_pbr_ooc_*`. Current SHA-256: reader
`da0b77324b703f45b53ff6d70932ef3e8857595a6c1afe1afc96827f68faea07`, wrapper
`70d9eb295172322f66dad3f5feeae34feec2066d4af8d903167165ab09133cf9`.
The earlier OOC/hash numbers below are historical, before PBR was added.

**MANUAL_REVIEW_REQUIRED.** `otpu_qst6_tmem_twopass` connects the experimental
16-word TMEM reader and two-pass INT6 row writer. For one row of 1..255 full
groups it reads the source twice, finds one FTZ maximum, quantizes and emits
three packed words per group and a scale. `busy_o` represents required source
ownership and remains asserted during reset recovery. Actual arbitration and
scoreboard enforcement of that ownership are external and not implemented.

The reader accepts TMEM's registered `frz` and waits while replay is pending.
The standalone IMPL=3 replay test uses two ports with conflicting reads and
prolonged replay; a version without capture freeze failed its assertion. It
also checks that reset blocks new reads until replay drains. The combined test
loads the production `otpu_tmem` RTL via its write port, forces a competing
read and replay, then compares six packed output words and one scale with the
Python INT6 reference. It checks invalid source/destination and reset during a
replay. The testbench caught the wrapper deasserting `busy` on reset; the
wrapper now retains it until recovery completes. The final selected series
of group-reader, replay, combined-path and two row-writer tests: **15 passed**.
An earlier selection without the combined-path test reported **13 passed,
1 failed** due to an Icarus-generated VVP syntax error; its repeat passed
**14**. These results do not establish the cause of the tool instability.

The original combined test used `IMPL=3`, `RPL=1`, `PMP=2` core-clock model and a single
competing TOPP port; this is not the board's PMP=1 2x-clock implementation or
the actual slice arbiter. XSim compiles the synthesizable TMEM path with
`SYNTHESIS` defined (the simulation-only TMEM assertions and initialization
are disabled), and the testbench initializes replay state. DRAM writes and
write-idle are a behavioral sink, not the real memory controller. The combined
test uses `-O0` because default XSim elaboration failed an LLVM assertion.
Another elaboration crashed with `0xC0000005`; a repeat reached the intended
reset assertion and the final correction passed. Tool causes are unknown.

OOC Vivado 2024.2.2, xc7k480tffg1156-2, 7.500 ns, exit 0: entire experimental
hierarchy **3472 LUT, 4229 FF, 18 DSP, 0 BRAM**, synthesis WNS +1.618 ns,
WHS +0.123 ns. No placement/route, full IO constraints or `HD.CLK_SRC`;
this is not the incremental cost or timing of the full r8. DRC reports CFGBVS
and DSP pipeline warnings. Reports: `build/logs/qst6_tmem_twopass_ooc_*`.
Source SHA-256: reader `8c80ba3bb5f06dac91eaa91b703f4df0eab715b02aafdebef97f219781d39b40`;
wrapper `6df75a879b76e9245cfc54b98c86f4b6f46cc884aba2d323b4de356b17968db7`.

Not implemented: production slice/ISA and capability integration; actual source
lock and dependency accounting; real shared
DDR write grant, fence, error and reset/partial-row invalidation; V-transpose
and partial groups; INT6 MXU read path and model numerical validation. Source
changes between passes can corrupt output. Mandatory lint/independent RTL
review, full regressions, full r8 synthesis/route and physical testing remain.
