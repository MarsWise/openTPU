"""Independent integer binary32 -> signed Q6 reference; no production code imports."""
import random
import struct
from pathlib import Path


def ref(bits):
    sign = bits >> 31
    exp = (bits >> 23) & 255
    frac = bits & 0x7FFFFF
    if exp == 255:
        return 1, 0  # unsupported non-finite input
    if exp < 126:
        return 0, 0  # includes FTZ, signed zero and |x| < 0.5
    if exp >= 132:
        magnitude = 31
    else:
        mant = (1 << 23) | frac
        shift = 150 - exp
        whole, rem = divmod(mant, 1 << shift)
        halfway = 1 << (shift - 1)
        magnitude = min(31, whole + (rem > halfway or (rem == halfway and whole & 1)))
    return 0, ((-magnitude if sign else magnitude) & 255)


def bits32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def main():
    values = [0, 0x80000000, 1, 0x007fffff, 0x00800000,
              0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001,
              0x7f7fffff, 0xff7fffff]
    for n in range(33):
        for sign in (0, 0x80000000):
            tie = bits32(n + 0.5) | sign
            values.extend((tie - 1, tie, tie + 1))
    rng = random.Random(0x165eed)
    values.extend(rng.getrandbits(32) for _ in range(10000))
    out = Path('build/qst6_round_vectors.txt')
    with out.open('w', encoding='ascii', newline='\n') as f:
        for bits in values:
            invalid, value = ref(bits)
            f.write(f'{bits:08x} {invalid:x} {value:02x}\n')
    print(f'{len(values)} vectors written to {out}')


if __name__ == '__main__':
    main()
