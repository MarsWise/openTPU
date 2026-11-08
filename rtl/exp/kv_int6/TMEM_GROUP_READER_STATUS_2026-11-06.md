# Experimental INT6 group reader against TMEM

**MANUAL_REVIEW_REQUIRED.** This status originally described the no-replay
reader; the replay-capable update and its scope are recorded below.
`otpu_kv6_tmem_group_reader` reads 16 contiguous
32-bit words as two eight-lane requests. It holds requests while grant is low,
captures the returned words in order, holds the assembled 512-bit group under
output backpressure and rejects a group outside `TMEM_WORDS`. Reset aborts
local work. The reader is now instantiated by an experimental wrapper with
the row encoder; neither is part of production QST or r8 firmware.

The reader relies on a dedicated port: a granted read returns the eight words
on the next cycle and the port is not stolen before capture. It does not
implement shared arbitration,
multi-slice coordination or source ownership during the two passes. The board
flow ordinarily selects **TMEM_IMPL=3 and RPL=1**, so this reader is **not
ready for production connection**. Although `t_frz_i` holds capture while a
replay is pending and eight-bank PBR outputs have been added, real slice
arbitration, physical pumped clocks and whole-system dependencies still need
integration and verification. See `TMEM_TWOPASS_STATUS_2026-11-06.md`.

## Replay update

`test_kv6_tmem_replay_xsim.py` uses actual `otpu_tmem` IMPL=3/RPL=1 with two
contending read ports, TOPP priority and PMP=2. Both halves encounter replay;
the reader waits through repeated losses, returns the expected 16 words, and
does not accept a new command after reset until an old replay drains. A version
without capture freeze produced `Fatal: reader advanced while replay pending`.
One elaboration crashed with `0xC0000005`; a repeated run passed. As below,
`SYNTHESIS` disables simulation-only TMEM assertions; the testbench initializes
the replay bookkeeping and supplies explicit checks. The test does not cover
all arbitration cases or board timing. A later combined test includes the
PMP=1 RTL on simulated 2x clocks and PBR, as documented in that status.

An experimental wrapper now connects this reader to the two-pass row encoder;
see `TMEM_TWOPASS_STATUS_2026-11-06.md`. The prior OOC resource/hash values
below describe the older reader without the freeze input, not the new source.

## Verification and limitations

`tests/test_kv6_tmem_group_xsim.py` connects the reader to the repository's
actual `otpu_tmem` RTL, using 256 words loaded through its write interface,
`IQ=1`, grant stalls, output stalls, unaligned bases, end-of-memory bounds and
an in-flight synchronous reset. It covers IMPL 0 and 1, plus IMPL 3 with
`PMP=2` core-clock RAM model, `RPL=0`, a single read port and no competing
readers. This is not the board's pumped clock, replay or slice arbiter.
The final selected run of the three reader configurations and the five existing
two-pass row tests reported **8 passed**; those tests execute the components
separately, not as one connected end-to-end RTL data path.

XSim rejected the original TMEM simulation-only initialization/debug hierarchy
and mixed timescales. The test compiles the synthesizable TMEM path with
`SYNTHESIS` defined, supplies an explicit timescale, initializes memory via its
write port and checks its own outcomes; TMEM's simulation-only assertions are
therefore absent. The first reset test also timed out because its testbench
did not hold reset over a rising edge; that stimulus was corrected. These are
not RTL regression passes and are recorded as setup/testbench failures.

Vivado 2024.2.2 OOC synthesis, xc7k480tffg1156-2, 7.500 ns, exit 0: 35 LUT,
548 FF, 0 DSP, 0 BRAM. WNS +5.743 ns and WHS +0.286 ns are synthesis-only
estimates without complete I/O and clock constraints, placement or route;
they do not represent the combined r8. Logs: `build/logs/kv6_tmem_group_ooc*`.
Reader SHA-256:
`70c6e8670c87c124d272e6f0b7a7470e0b66a9e4dca7227ef0a09728a242d30d`.

Mandatory `bin/rtl-lint`/slang and independent review are unavailable; there
is no formal protocol/CDC proof, real DDR interaction, full regression, full
implementation or board/model test. Do not treat OOC timing as signoff.
