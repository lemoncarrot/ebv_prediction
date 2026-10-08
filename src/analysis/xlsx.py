"""Minimal .xlsx -> DataFrame with the standard library (no openpyxl)."""
import re, sys, zipfile, xml.etree.ElementTree as ET
import pandas as pd
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group()
    n = 0
    for ch in letters: n = n * 26 + ord(ch) - 64
    return n - 1

def read_xlsx(path, sheet=1):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        root = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))
    rows = []
    for r in root.iter(f"{{{NS['m']}}}row"):
        row = {}
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS); t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
            elif v is None: continue
            elif t == "s": val = shared[int(v.text)]
            else: val = v.text
            row[col_index(c.get("r"))] = val
        rows.append(row)
    width = max(max(r) for r in rows if r) + 1
    grid = [[r.get(i) for i in range(width)] for r in rows]
    df = pd.DataFrame(grid[1:], columns=[str(h) for h in grid[0]])
    for c in df.columns:  # numeric where every value parses
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() == df[c].notna().sum():
            df[c] = converted
    return df

