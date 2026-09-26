"""Find and decompress the section pages of an R2004+ (AC1018..AC1032) DWG file, without libredwg.
Ported from libredwg's decode.c (decompress_R2004_section and the encrypted page header)."""
import struct, sys, collections

def decompress(src, out_size):
    dst = bytearray(); pos = 0; n = len(src)
    def rd():
        nonlocal pos
        b = src[pos]; pos += 1; return b
    def literal_length():
        b = rd()
        if 1 <= b <= 0x0F: return b + 3, 0
        if b == 0:
            total = 0x0F
            while True:
                b = rd()
                if b != 0: break
                total += 0xFF
            return total + b + 3, 0
        if b & 0xF0: return 0, b
        return 0, 0
    def two_byte_offset():
        a = rd(); b = rd()
        return (a >> 2) | (b << 6), a & 3
    def long_offset():
        b = rd(); total = 0
        if b == 0:
            total = 0xFF
            while True:
                b = rd()
                if b != 0: break
                total += 0xFF
        return total + b
    lit, _ = literal_length()
    dst += src[pos:pos + lit]; pos += lit
    op = 0
    while pos < n and len(dst) < out_size:
        if op == 0: op = rd()
        if op >= 0x40:
            comp = ((op & 0xF0) >> 4) - 1; op2 = rd(); off = (op2 << 2) | ((op & 0x0C) >> 2)
            if op & 3: lit = op & 3; op = 0
            else: lit, op = literal_length()
        elif 0x21 <= op <= 0x3F:
            comp = op - 0x1E; off, lit = two_byte_offset()
            if lit: op = 0
            else: lit, op = literal_length()
        elif op == 0x20:
            comp = long_offset() + 0x21; off, lit = two_byte_offset()
            if lit: op = 0
            else: lit, op = literal_length()
        elif 0x12 <= op <= 0x1F:
            comp = (op & 0x0F) + 2; off, lit = two_byte_offset(); off += 0x3FFF
            if lit: op = 0
            else: lit, op = literal_length()
        elif op == 0x10:
            comp = long_offset() + 9; off, lit = two_byte_offset(); off += 0x3FFF
            if lit: op = 0
            else: lit, op = literal_length()
        elif op == 0x11: break
        else: raise ValueError('bad opcode %#x at %d' % (op, pos))
        s = len(dst) - off - 1
        if s < 0: raise ValueError('offset underflow')
        for _ in range(comp):
            dst.append(dst[s]); s += 1
        if lit:
            dst += src[pos:pos + lit]; pos += lit
    return bytes(dst)

def find_pages(data):
    """Every 32-byte-aligned offset whose scrambled header decodes to the data-page tag."""
    pages = []
    for a in range(0x100, len(data) - 32, 0x20):
        w0 = struct.unpack_from('<I', data, a)[0] ^ (0x4164536b ^ a)
        if w0 == 0x4163043b:
            f = [struct.unpack_from('<I', data, a + 4 * k)[0] ^ (0x4164536b ^ a) for k in range(8)]
            pages.append(dict(addr=a, type=f[1], data_size=f[2], section_size=f[3], start=f[4], f5=f[5], f6=f[6], f7=f[7]))
    return pages

if __name__ == '__main__':
    data = open(sys.argv[1], 'rb').read()
    pages = find_pages(data)
    print(len(pages), 'data pages found')
    print(collections.Counter(p['type'] for p in pages).most_common(8))
    for p in pages[:4]: print(p)
