# Reading the 3D solids out of a 2013+ DWG

`libredwg` (and so `dwg2dxf`) leaves the 3D solids of an AutoCAD 2013-2018 DWG empty: they live in the file's
"AcDb:AcDsPrototype_1b" data storage section, compressed. These two scripts read them without CAD software.

* `dwg_pages.py` finds every data page of the file (32-byte header scrambled with `0x4164536b ^ file offset`,
  tag `0x4163043b`), and decompresses it with a port of libredwg's R2004 LZ77 decoder. The pages with section
  type 13 are the data storage section; sorted by their start offset they join into one stream.
* `asm_load.py` cuts that stream into the `ASM BinaryFile4` blocks (Autodesk ShapeManager, a variant of ACIS
  SAB) and loads them with `ezdxf.acis`, after patching two things ezdxf does not know: the end-of-data
  padding and this version's transform record.

```
python dwg_pages.py file.dwg                     # counts the pages
# then, in Python: join the type-13 pages (see the notes in asm_load.py) into acds.bin and
bodies = asm_load.load_all("acds.bin")           # ezdxf.acis bodies; ezdxf.acis.api.mesh_from_body(body)
```

`ezdxf` meshes only the flat faces; curved faces (fillets, cylinders such as door handles, most furniture) are
skipped. The CAD data itself is the manufacturer's and is not in this repository.
