"""Dev helper: plumb a new CAPS2 capability bit next to the last one (PREV), as TGRQ was.
usage: python addcap.py REPO NAME BIT PREV "description"
"""
import re
import sys
from pathlib import Path

root, N, bit, P, desc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
n, p = N.lower(), P.lower()


def sub(rel, old, new, count=1):
    f = root / rel
    s = f.read_text()
    k = s.count(old)
    if k != count:
        raise SystemExit(f"{rel}: {k} x {old!r}")
    f.write_text(s.replace(old, new))


def subre(rel, pat, rep, count=1):
    f = root / rel
    s = f.read_text()
    s2, k = re.subn(pat, rep, s)
    if k != count:
        raise SystemExit(f"{rel}: {k} x /{pat}/")
    f.write_text(s2)


# python side
sub("opentpu/host/regs.py", f'"{p}": bool(v & CAP2_{P})}}',
    f'"{p}": bool(v & CAP2_{P}),\n            "{n}": bool(v & CAP2_{N})}}')
sub("opentpu/host/regs.py", f'"{p}": False}}', f'"{p}": False, "{n}": False}}')
subre("opentpu/host/regs.py", rf"(CAP2_{P} = \d+ +#[^\n]*\n)",
      rf"\g<1>CAP2_{N} = {1 << bit}{' ' * max(1, 9 - len(str(1 << bit)))}# {desc}\n")
sub("opentpu/host/diag.py", f"R.CAP2_{P})", f"R.CAP2_{P} | R.CAP2_{N})")
sub("opentpu/host/fake.py", f"{p}: bool = False) -> dict:", f"{p}: bool = False, {n}: bool = False) -> dict:")
sub("opentpu/host/fake.py", f'"{P}": {p}}})', f'"{P}": {p}, "{N}": {n}}})')
sub("opentpu/host/fake.py", f"{p}: bool = False):", f"{p}: bool = False, {n}: bool = False):")
sub("opentpu/host/fake.py", f"{p}))", f"{p}, {n}))")
subre("opentpu/host/fake.py", rf"(        self\.{p} = {p} +#[^\n]*\n)",
      rf"\g<1>        self.{n} = {n}{' ' * max(1, 16 - len(n) - len(n))}# CAPS2 bit{bit}: {desc}\n")
sub("opentpu/host/fake.py", f"(R.CAP2_{P} if self.{p} else 0)",
    f"(R.CAP2_{P} if self.{p} else 0) \\\n                | (R.CAP2_{N} if self.{n} else 0)")
sub("opentpu/host/board.py", f'{P}=p.get("{P}", 0))))', f'{P}=p.get("{P}", 0), {N}=p.get("{N}", 0))))')
sub("opentpu/host/board.py", f'    {p} = bool((info.get("caps2") or {{}}).get("{p}"))\n',
    f'    {p} = bool((info.get("caps2") or {{}}).get("{p}"))\n'
    f'    {n} = bool((info.get("caps2") or {{}}).get("{n}"))\n')
subre("opentpu/host/board.py", rf'(\("{P}", {p}, "[^"]*"\))\):',
      rf'\g<1>,\n                          ("{N}", {n}, "{desc} (CAPS2 bit{bit})")):')
sub("opentpu/host/board.py", f'"{P}": {p},', f'"{P}": {p}, "{N}": {n},')
sub("opentpu/host/board.py", f'{P}=bool((info.get("caps2") or {{}}).get("{p}")),',
    f'{P}=bool((info.get("caps2") or {{}}).get("{p}")),\n'
    f'                           {N}=bool((info.get("caps2") or {{}}).get("{n}")),')
sub("opentpu/target.py", f'"{P}"):\n', f'"{P}", "{N}"):\n')
sub("opentpu/target.py", f'        "{P}": int(a.get("{P}", False)),\n',
    f'        "{P}": int(a.get("{P}", False)),\n        "{N}": int(a.get("{N}", False)),\n')
sub("opentpu/target.py", f'    if knobs.get("{P}") is not None:\n        a["{P}"] = bool(int(knobs["{P}"]))\n',
    f'    if knobs.get("{P}") is not None:\n        a["{P}"] = bool(int(knobs["{P}"]))\n'
    f'    if knobs.get("{N}") is not None:\n        a["{N}"] = bool(int(knobs["{N}"]))\n')
for t in ("bcu1525", "nangate45-small", "nangate45", "ypcb-00338"):
    subre(f"targets/{t}.toml", rf"({P} = false[^\n]*\n)",
          rf"\g<1>{N} = false{' ' * max(1, 26 - len(N))}# {desc} (CAPS2 bit{bit})\n")
# builds
sub("boards/ypcb-00338/run_vivado.sh", f'[[ "${p}" == false ]] && {p}=0\n',
    f'[[ "${p}" == false ]] && {p}=0\n{n}="${{{N}:-${{T_{N}:-0}}}}"\n'
    f'[[ "${n}" == true ]] && {n}=1; [[ "${n}" == false ]] && {n}=0\n')
sub("boards/ypcb-00338/run_vivado.sh", f'{P}="${p}" 2>/dev/null', f'{P}="${p}" {N}="${n}" 2>/dev/null')
sub("boards/ypcb-00338/run_vivado.sh", f'export OTPU_{P}_GEN="${p}"\n',
    f'export OTPU_{P}_GEN="${p}"\nexport OTPU_{N}_GEN="${n}"\n')
sub("boards/ypcb-00338/vivado/create_project.tcl",
    f'set {P} [expr {{[info exists ::env(OTPU_{P}_GEN)] ? $::env(OTPU_{P}_GEN) eq "1" : 0}}]\n',
    f'set {P} [expr {{[info exists ::env(OTPU_{P}_GEN)] ? $::env(OTPU_{P}_GEN) eq "1" : 0}}]\n'
    f'set {N} [expr {{[info exists ::env(OTPU_{N}_GEN)] ? $::env(OTPU_{N}_GEN) eq "1" : 0}}]\n')
sub("boards/ypcb-00338/vivado/create_project.tcl", f"{P}=1'b${P} ", f"{P}=1'b${P} {N}=1'b${N} ")
# RTL
sub("rtl/shell/otpu_board.sv", f".{P}({P}),\n", f".{P}({P}), .{N}({N}),\n")
subre("rtl/shell/otpu_board.sv", rf"(  parameter bit {P} += 1'b0,[^\n]*\n)",
      rf"\g<1>  parameter bit {N}{' ' * max(1, 12 - len(N))}= 1'b0,    // {desc} (CAPS2 bit{bit})\n")
subre("rtl/shell/otpu_ctrl.sv", rf"(  parameter bit {P} = 1'b0,[^\n]*\n)",
      rf"\g<1>  parameter bit {N} = 1'b0,{' ' * max(1, 13 - len(N))}// CAPS2 bit{bit}: {desc}\n")
subre("rtl/shell/otpu_ctrl.sv", rf"CAPS2 = \{{(\d+)'d0, {P},",
      lambda m: f"CAPS2 = {{{int(m.group(1)) - 1}'d0, {N}, {P},")
for f in ("rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv", "rtl/boards/ypcb-00338/otpu_native_sys.sv"):
    sub(f, f".{P}({P})) ", f".{P}({P}), .{N}({N})) ")
subre("rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv", rf"(  parameter bit {P} *= 1'b0)( +)(//[^\n]*\n)",
      rf"\g<1>,\2\3  parameter bit {N} = 1'b0                // {desc} (CAPS2 bit{bit})\n")
sub("rtl/boards/ypcb-00338/otpu_native_sys.sv", f"  parameter bit {P} = 1'b0\n",
    f"  parameter bit {P} = 1'b0,\n  parameter bit {N} = 1'b0\n")
subre("sim/verilator/tb_board.sv", rf"(  parameter bit {P} *= 1'b0;\n)",
      rf"\g<1>  parameter bit {N}{' ' * max(1, 10 - len(N))}= 1'b0;\n")
sub("sim/verilator/tb_board.sv", f".{P}({P}), .ARCH_HASH", f".{P}({P}), .{N}({N}), .ARCH_HASH")
print("ok", N, bit)
