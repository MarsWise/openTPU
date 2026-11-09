"""A smaller Dory GGUF from the real one, for the container's real-weight check (tests/test_dory.py):
physical blocks 0..7 (input), 16..23 (recurrent), 56..63 (output) renumbered 0..23, each part
counted as 8 blocks (4 layers), the pattern / RoPE profile cut alike, the embedding / head /
logit scales cut to the first V vocabulary rows (tokenizer arrays too, merges dropped). Tensor
bytes copied as they are. Resumable (a call stops after 140 s; run it again).

    python tools/dory_subset_gguf.py NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf dory_sub24.gguf 32768
"""
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
V = int(sys.argv[3])
groups = [(0, 8), (16, 24), (56, 64)]            # input / recurrent / output physical ranges
keep = [b for a, z in groups for b in range(a, z)]
remap = {b: i for i, b in enumerate(keep)}
f = open(src, 'rb')
def rd(n): return f.read(n)
def u32(): return struct.unpack('<I', rd(4))[0]
def u64(): return struct.unpack('<Q', rd(8))[0]
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip_val(t):
    if t == 8: n = u64(); return rd(n)
    if t == 9:
        et = u32(); n = u64(); out = []
        for _ in range(n): out.append(skip_val(et))
        return (et, out)
    return rd(SZ[t])
def enc_str(s): b = s.encode(); return struct.pack('<Q', len(b)) + b
def enc_val(t, v):
    if t == 8: return struct.pack('<Q', len(v)) + v
    if t == 9:
        et, items = v; return struct.pack('<IQ', et, len(items)) + b''.join(enc_val(et, i) for i in items)
    return v
assert rd(4) == b'GGUF'; ver = u32(); nt = u64(); nk = u64()
kvs = []
for _ in range(nk):
    n = u64(); k = rd(n).decode(); t = u32(); v = skip_val(t); kvs.append((k, t, v))
tis = []
for _ in range(nt):
    n = u64(); name = rd(n).decode(); nd = u32(); dims = [u64() for _ in range(nd)]; tt = u32(); off = u64()
    tis.append((name, dims, tt, off))
align = 32
for k, t, v in kvs:
    if k == 'general.alignment': align = struct.unpack('<I', v)[0]
pos = f.tell(); data0 = (pos + align - 1) // align * align
BLK = {30: (1, 2), 13: (256, 176), 14: (256, 210), 0: (1, 4), 12: (256, 144), 1: (1, 2), 8: (32, 34)}
def nbytes(dims, tt):
    be, bb = BLK[tt]; n = 1
    for d in dims: n *= d
    return n // be * bb
new = []
for name, dims, tt, off in tis:
    if name.startswith('blk.'):
        b = int(name.split('.')[1])
        if b not in remap: continue
        nn = 'blk.%d.' % remap[b] + name.split('.', 2)[2]
        new.append((nn, dims, tt, off, nbytes(dims, tt)))
    elif name in ('token_embd.weight', 'output.weight'):
        d2 = [dims[0], V]; new.append((name, d2, tt, off, nbytes(d2, tt)))
    elif name == 'output_scale.weight':
        d2 = [V]; new.append((name, d2, tt, off, nbytes(d2, tt)))
    else:
        new.append((name, dims, tt, off, nbytes(dims, tt)))
L = len(keep)
mod = {'dory.block_count': struct.pack('<I', L), 'dory.input_layer_count': struct.pack('<I', 8),
       'dory.recurrent_layer_count': struct.pack('<I', 8), 'dory.output_layer_count': struct.pack('<I', 8)}
out = []
for k, t, v in kvs:
    if k in mod: v = mod[k]
    if k == 'dory.layer_pattern': v = ('S-S-S-*-' * (L // 8)).encode()
    if k == 'dory.rope.profile':
        et, items = v; v = (et, [items[b] for b in keep])
    if k.startswith('tokenizer.ggml.') and t == 9:
        et, items = v; v = (et, items[:V]) if len(items) > V else v
        if k == 'tokenizer.ggml.merges': v = (et, [])
    out.append((k, t, v))
import os, time
t0 = time.time()

class W:
    def __init__(s): s.b = []
    def write(s, x): s.b.append(x)
    def tell(s): return sum(map(len, s.b))
g = W()
g.write(b'GGUF' + struct.pack('<IQQ', ver, len(new), len(out)))
for k, t, v in out: g.write(enc_str(k) + struct.pack('<I', t) + enc_val(t, v))
o = 0; offs = []
for name, dims, tt, _, nb in new:
    offs.append(o); g.write(enc_str(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', tt, o))
    o = (o + nb + align - 1) // align * align
p = g.tell(); g.write(b'\0' * ((p + align - 1) // align * align - p))
head = b''.join(g.b); base = len(head)
if not os.path.exists(dst) or os.path.getsize(dst) < base:
    open(dst, 'wb').write(head)
total = base + o
with open(dst, 'r+b') as h:
    for (name, dims, tt, off, nb), no in zip(new, offs):
        a = base + no; e = a + nb
        cur = os.path.getsize(dst)
        if e <= cur: continue
        st = max(a, cur)
        h.seek(st); f.seek(data0 + off + (st - a))
        while st < e:
            if time.time() - t0 > 140: print('more', st, total); sys.exit(0)
            n = min(e - st, 16 << 20); h.write(f.read(n)); st += n; h.flush()
        pad = (e + align - 1) // align * align - e
        h.write(b'\0' * pad); h.flush()
print('done', os.path.getsize(dst), total)
