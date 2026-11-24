#!/bin/bash
# Claude: what the new models' files hold, read without running their code (no card use):
# Maple-Preview's GGUF header (metadata, every tensor's name, shape and type) and Ivonar Nano's
# packed checkpoint (torch.load weights_only: tensors and containers only; keys, shapes, dtypes),
# its README's architecture notes. The log is what maple.py / ivonar.py are written against.
set -u
M=$HOME/openTPU_BitNET/models
P=$HOME/hfref/bin/python; "$P" -c 'import torch' 2>/dev/null || P=$HOME/openTPU_BitNET/.venv/bin/python
echo "== $(date +%T) Maple-Preview GGUF"
"$P" -I - "$M/maple-preview/maple-preview-TQ2_0-head-F16.gguf" <<'PY'
import struct, sys, collections
f = open(sys.argv[1], 'rb')
u32 = lambda: struct.unpack('<I', f.read(4))[0]
u64 = lambda: struct.unpack('<Q', f.read(8))[0]
def s(): n = u64(); return f.read(n).decode('utf8', 'replace')
assert f.read(4) == b'GGUF'; ver = u32(); nt = u64(); nk = u64()
print('version', ver, 'tensors', nt, 'kv', nk)
TS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
for _ in range(nk):
    k = s(); t = u32()
    if t == 8: v = s()[:300]
    elif t == 9:
        et = u32(); n = u64()
        if et == 8: v = [s() for _ in range(n)]; v = f'<{n} strings> {v[:5]}'
        else:
            raw = f.read(TS[et] * n)
            v = f'<{n} x type {et}> ' + (str(list(struct.unpack("<%d%s" % (min(n, 16), FMT[et][1]), raw[:TS[et] * min(n, 16)]))) if n else '')
    else: v = struct.unpack(FMT[t], f.read(TS[t]))[0]
    if 'token' in k and isinstance(v, str) and len(v) > 200: v = v[:200]
    print('KV', k, '=', v)
types = collections.Counter(); rows = []
for _ in range(nt):
    n = s(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
    rows.append((n, dims, ty, off)); types[ty] += 1
print('tensor types', dict(types))
for n, d, t, o in rows:
    if '.' in n and n.split('.')[0] == 'blk' and int(n.split('.')[1]) > 1:
        continue
    print('T', n, d, t, o)
print('layers', sorted({int(n.split('.')[1]) for n, *_ in rows if n.startswith('blk.')})[-1] + 1)
PY
echo; echo "== $(date +%T) Maple-Preview config / README"
cat "$M/maple-preview/config.json"; echo; sed -n 1,80p "$M/maple-preview/README.md"
echo; echo "== $(date +%T) Ivonar Nano"
sed -n 1,200p "$M/ivonar-nano/README.md"
"$P" -I - "$M/ivonar-nano/packed_inference_checkpoint.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
def walk(o, p='', depth=0):
    if isinstance(o, torch.Tensor):
        print('T', p, tuple(o.shape), o.dtype, float(o.float().abs().mean()) if o.numel() else 0); return
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, f'{p}.{k}' if p else str(k), depth + 1)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o[:64]): walk(v, f'{p}[{i}]', depth + 1)
    else:
        print('V', p, repr(o)[:300])
walk(ck)
PY
echo "== $(date +%T) DONE"
