#!/usr/bin/env python3
"""Turn a TrueType/OpenType font into the small JSON makecar's text engine reads.
Usage (from the repo root): python -m scripts.extract_font DejaVuSans-Bold.ttf -o makecar/data/fonts/sans-bold.json --name sans-bold

Needs fonttools (pip install fonttools); the JSON itself needs nothing.  Curves are
flattened to short polylines in font units, so a glyph is just a list of closed
contours.  Kerning comes from the `kern` table and GPOS pair positioning.
"""
import argparse
import json
from pathlib import Path

from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

CHARS = [chr(c) for c in range(32, 127)] + [chr(c) for c in range(160, 256)] + list("–—‘’“”•€")


class FlattenPen(BasePen):
    """Records closed contours as point lists; curves become polylines."""

    def __init__(self, glyph_set, segment_units: float):
        super().__init__(glyph_set)
        self.contours = []
        self._cur = None
        self._seg = segment_units

    def _moveTo(self, p):
        self._cur = [p]

    def _lineTo(self, p):
        self._cur.append(p)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._cur[-1]
        length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in ((p0, p1), (p1, p2), (p2, p3)))
        n = max(2, min(12, int(length / self._seg) + 1))
        for k in range(1, n + 1):
            t = k / n
            u = 1 - t
            x = u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0]
            y = u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1]
            self._cur.append((x, y))

    def _closePath(self):
        if self._cur and len(self._cur) >= 3:
            if self._cur[0] == self._cur[-1]:
                self._cur.pop()
            self.contours.append([[round(x), round(y)] for x, y in self._cur])
        self._cur = None

    def _endPath(self):
        self._closePath()


def gpos_kerning(font: TTFont, names: dict) -> dict:
    """Pair kerning (in font units) for every pair of our characters, summed over GPOS pair lookups."""
    out: dict = {}
    if "GPOS" not in font:
        return out
    gpos = font["GPOS"].table
    lookups = gpos.LookupList.Lookup
    glyph_to_char = {}
    for ch, g in names.items():
        glyph_to_char.setdefault(g, ch)
    pair_subtables = []
    for lookup in lookups:
        for st in lookup.SubTable:
            if lookup.LookupType == 9 and st.ExtensionLookupType == 2:
                st = st.ExtSubTable
            elif lookup.LookupType != 2:
                continue
            pair_subtables.append(st)
    for st in pair_subtables:
        coverage = list(st.Coverage.glyphs)
        if st.Format == 1:
            for first, pair_set in zip(coverage, st.PairSet):
                if first not in glyph_to_char:
                    continue
                for rec in pair_set.PairValueRecord:
                    second = rec.SecondGlyph
                    if second in glyph_to_char and rec.Value1 is not None:
                        v = getattr(rec.Value1, "XAdvance", 0) or 0
                        if v:
                            key = glyph_to_char[first] + glyph_to_char[second]
                            out[key] = out.get(key, 0) + v
        elif st.Format == 2:
            cd1, cd2 = st.ClassDef1.classDefs, st.ClassDef2.classDefs
            cov = set(coverage)
            for first, ch1 in glyph_to_char.items():
                if first not in cov:
                    continue
                c1 = cd1.get(first, 0)
                if c1 >= len(st.Class1Record):
                    continue
                rec1 = st.Class1Record[c1]
                for second, ch2 in glyph_to_char.items():
                    c2 = cd2.get(second, 0)
                    if c2 >= len(rec1.Class2Record):
                        continue
                    val = rec1.Class2Record[c2].Value1
                    v = (getattr(val, "XAdvance", 0) or 0) if val is not None else 0
                    if v:
                        out[ch1 + ch2] = out.get(ch1 + ch2, 0) + v
    return out


def extract(path: Path, name: str, chars=CHARS) -> dict:
    font = TTFont(str(path))
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    hmtx = font["hmtx"].metrics
    os2 = font["OS/2"]
    hhea = font["hhea"]
    names = {ch: cmap[ord(ch)] for ch in chars if ord(ch) in cmap}
    glyphs = {}
    for ch, gname in names.items():
        pen = FlattenPen(glyph_set, segment_units=upm * 0.035)
        glyph_set[gname].draw(pen)
        glyphs[ch] = {"adv": hmtx[gname][0], "contours": pen.contours}
    kerning = gpos_kerning(font, names)
    if "kern" in font:
        for table in font["kern"].kernTables:
            kt = getattr(table, "kernTable", {})
            glyph_to_char = {g: c for c, g in names.items()}
            for (a, b), v in kt.items():
                if a in glyph_to_char and b in glyph_to_char and v:
                    key = glyph_to_char[a] + glyph_to_char[b]
                    kerning[key] = kerning.get(key, 0) + v
    cap = getattr(os2, "sCapHeight", 0) or 0
    if not cap and "H" in glyphs:
        cap = max(p[1] for c in glyphs["H"]["contours"] for p in c)
    xh = getattr(os2, "sxHeight", 0) or 0
    if not xh and "x" in glyphs:
        xh = max(p[1] for c in glyphs["x"]["contours"] for p in c)
    family = font["name"].getDebugName(1) or path.stem
    style = font["name"].getDebugName(2) or ""
    return {
        "name": name, "source": f"{family} {style}".strip(), "units_per_em": upm,
        "ascender": hhea.ascent, "descender": hhea.descent, "cap_height": cap, "x_height": xh,
        "glyphs": glyphs, "kerning": kerning,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("font")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", help="font name used in configs (default: the output file stem)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    data = extract(Path(args.font), args.name or out.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{data['source']}: {len(data['glyphs'])} glyphs, {len(data['kerning'])} kerning pairs -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
