#!/bin/bash
# Claude: lists the processes on the host that hold the card (the run lock, the xdma0 lock or /dev/xdma0*)
# and, with the argument "kill", stops them (TERM, then KILL after 5 s). No bitstream load, no rescan,
# no register writes: a stopped run only leaves its own DRAM contents behind.
pids() {
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    ls -l "$p/fd" 2>/dev/null | grep -qE 'claude_run\.lock|otpu/xdma0\.lock|/dev/xdma0' && echo "${p#/proc/}"
  done
}
P=$(pids)
[ -z "$P" ] && { echo "== the card is free: nothing is running"; exit 0; }
echo "== runs holding the card:"
for p in $P; do ps -o pid=,etime=,cmd= -p "$p" | cut -c1-160; done
[ "${1:-}" = kill ] || exit 3
kill $P 2>/dev/null; sleep 5
P=$(pids); [ -n "$P" ] && { kill -9 $P 2>/dev/null; sleep 1; }
P=$(pids); [ -z "$P" ] && echo "== stopped: the card is free" || { echo "== still running: $P"; exit 1; }
