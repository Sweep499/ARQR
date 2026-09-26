"""Load the ASM (Autodesk ShapeManager) solids that libredwg cannot read out of a DWG's data storage section."""
import re, collections
from ezdxf.acis import sab, api as acis
import numpy as np

_END = ('End-of-ASM-data', 'End-of-ACIS-data')
def _read_records_until_end(self):
    while True:
        if not self.has_data: return
        try: rec = self.read_record()
        except IndexError: return
        yield rec
        if rec and getattr(rec[0], 'value', None) in _END: return   # stop before the padding that follows
sab.Decoder.read_records = _read_records_until_end

# ASM 226 stores a transform as 4 vectors (3 rotation rows + translation), a scale and flags, not as a text
# string like ACIS 700 does.
from ezdxf.acis.const import Tags
def _read_transform(self):
    d, i = self.data, self.index
    rows = []
    for _ in range(4):
        rows.append(d[i].value); i += 1
    scale = d[i].value if d[i].tag == Tags.DOUBLE else 1.0
    i += 1
    while i < len(d) and d[i].tag in (Tags.BOOL_TRUE, Tags.BOOL_FALSE, Tags.STR, Tags.LITERAL_STR):
        i += 1
    self.index = i
    out = []
    for r in rows[:3]:
        out += [r[0] * scale, r[1] * scale, r[2] * scale]
    out += [rows[3][0], rows[3][1], rows[3][2]]
    return out
sab.SabDataLoader.read_transform = _read_transform

_END_MARK = b'End\x0e\x02of\x0e\x03ASM\r\x04data'      # the end-of-data record, as SAB writes it
def blobs(stream):
    starts = [m.start() for m in re.finditer(b'ASM BinaryFile4', stream)] + [len(stream)]
    out = []
    for a, b in zip(starts, starts[1:]):
        blob = stream[a:b]
        k = blob.rfind(_END_MARK)
        out.append((a, blob[:k + len(_END_MARK)] if k >= 0 else blob))   # drop the record padding after the marker
    return out

def load_all(stream_path):
    stream = open(stream_path, 'rb').read()
    out, errs = [], collections.Counter()
    for a, blob in blobs(stream):
        try:
            out.append((a, acis.load(blob)))
        except Exception as e:
            errs[type(e).__name__ + ': ' + str(e)[:90]] += 1
    return out, errs
