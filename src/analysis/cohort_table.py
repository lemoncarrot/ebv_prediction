#!/usr/bin/env python3
"""Cohort descriptor tables ("Table 1" style), one per cohort, split by EBV status.

    python cohort_table.py --out-dir ../../results/lab_meeting_2026-10

Describes exactly the patients analysed: slides in a run's splits.csv (the same
436 usable slides in every TCGA/Chile run), collapsed to one row per patient.
  TCGA:  STAD_merged_table.csv (age, gender, molecular subtype, MSI, histological
         type, pathologic stage); EBV status = molecular subtype STAD_EBV.
  Chile: Chile set_Training_n_132.xlsx (age at diagnosis, sex, Lauren type, grade,
         pTNM); EBV status = EBER-ISH. Joined on idscanner == sample_id.
Columns: Overall | EBV+ | EBV- | p. Continuous: median [IQR], Mann-Whitney U.
Categorical: n (%) within column, Fisher exact for 2x2 else chi-square; missing
values are shown but excluded from percentages' denominators and from tests.
Writes cohort_table_{tcga,chile}.csv / .png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlsx import read_xlsx  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent.parent
INK, SECONDARY, RULE = "#0b0b0b", "#52514e", "#c3c2b7"

TCGA_SUBTYPE = {"STAD_CIN": "CIN", "STAD_GS": "GS", "STAD_MSI": "MSI", "STAD_EBV": "EBV",
                "STAD_POLE": "POLE"}


def tcga_lauren(histology: str | float) -> str | float:
    """Approximate Lauren type from TCGA histological_type."""
    if not isinstance(histology, str):
        return np.nan
    h = histology.lower()
    if "diffuse" in h or "signet" in h:
        return "Diffuse"
    if "intestinal" in h:
        return "Intestinal"
    return "Adenocarcinoma NOS"


def chile_stage_group(ptnm: str | float) -> str | float:
    if not isinstance(ptnm, str):
        return np.nan
    return "III/IV" if ptnm.startswith(("III", "IV")) else "I/II"


def analysed(labels: pd.DataFrame, splits_csv: Path, cohort: str) -> pd.DataFrame:
    """One row per analysed patient: patient_id, sample_id, n_slides, EBV status."""
    used = set(pd.read_csv(splits_csv)["slide_id"])
    rows = labels[(labels["cohort"] == cohort) & labels["slide_id"].isin(used)]
    return (rows.groupby("patient_id", as_index=False)
            .agg(sample_id=("sample_id", "first"), n_slides=("slide_id", "size"),
                 ebv=("label", "max")))


def tcga_patients(labels: pd.DataFrame, splits_csv: Path) -> pd.DataFrame:
    pts = analysed(labels, splits_csv, "tcga")
    merged = (pd.read_csv(PROJECT / "metadata/STAD_merged_table.csv")
              .drop_duplicates("Patient.ID")
              .rename(columns={"Patient.ID": "patient_id"}))
    pts = pts.merge(merged, on="patient_id", how="left")
    return pd.DataFrame({
        "ebv": pts["ebv"], "n_slides": pts["n_slides"], "Age (years)": pts["Age"],
        "Sex": pts["Gender"].map({"M": "Male", "F": "Female"}),
        "Molecular subtype": pts["Subtype"].map(TCGA_SUBTYPE),
        "MSI status": pts["MSI"].map({"MSI": "MSI-high", "MSS": "MSS"}),
        "Lauren type (from histological type)": pts["histological_type"].map(tcga_lauren),
        "Pathologic stage": pts["P_stage"],
    })


def chile_patients(labels: pd.DataFrame, splits_csv: Path) -> pd.DataFrame:
    pts = analysed(labels, splits_csv, "chile")
    clin = read_xlsx(PROJECT / "data/original_chile/Chile set_Training_n_132.xlsx")
    pts = pts.merge(clin, left_on="sample_id", right_on="idscanner", how="left")
    unmatched = int(pts["idscanner"].isna().sum())
    if unmatched:
        print(f"WARNING: {unmatched} Chile patients have no clinical row")
    disagree = int((pts["eber_ish"].map({"Positive": 1, "Negative": 0}) != pts["ebv"]).sum())
    if disagree:
        print(f"WARNING: EBV label differs from spreadsheet EBER-ISH for {disagree} patients")
    return pd.DataFrame({
        "ebv": pts["ebv"], "n_slides": pts["n_slides"], "Age (years)": pts["age_diagnosis"],
        "Sex": pts["sex"],
        "Lauren type": pts["lauren_classification_report"].replace(
            {"Intestinal type": "Intestinal", "Diffuse type": "Diffuse",
             "Cannot be determined": "Undetermined"}),
        "Histologic grade": pts["histologic_grade"].str.extract(r"^(G\d)")[0],
        "Pathologic stage": pts["ptnm"].map(chile_stage_group),
    })


def pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def p_format(p: float) -> str:
    return "" if p is None or np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def build_table(pts: pd.DataFrame, orders: dict[str, list[str]],
                skip_test: set[str]) -> pd.DataFrame:
    groups = {"Overall": pts, "EBV+": pts[pts["ebv"] == 1], "EBV−": pts[pts["ebv"] == 0]}
    rows = [{"Characteristic": "Patients, n", **{k: str(len(g)) for k, g in groups.items()},
             "p": ""},
            {"Characteristic": "Slides, n", **{k: str(int(g["n_slides"].sum()))
                                                for k, g in groups.items()}, "p": ""}]
    age = "Age (years)"
    p = mannwhitneyu(groups["EBV+"][age].dropna(), groups["EBV−"][age].dropna()).pvalue
    rows.append({"Characteristic": f"{age}, median [IQR]", "p": p_format(p),
                 **{k: f"{g[age].median():.0f} [{g[age].quantile(.25):.0f}–"
                       f"{g[age].quantile(.75):.0f}]" for k, g in groups.items()}})
    if pts[age].isna().any():
        rows.append({"Characteristic": "   Missing", "p": "",
                     **{k: str(int(g[age].isna().sum())) for k, g in groups.items()}})

    for var, levels in orders.items():
        known = pts[pts[var].notna()]
        table = pd.crosstab(known[var], known["ebv"]).reindex(levels).fillna(0)
        table = table.loc[table.sum(axis=1) > 0]
        p = np.nan
        if var not in skip_test and table.shape[0] > 1 and table.shape[1] == 2:
            p = (fisher_exact(table.values)[1] if table.shape == (2, 2)
                 else chi2_contingency(table.values)[1])
        rows.append({"Characteristic": var, "Overall": "", "EBV+": "", "EBV−": "",
                     "p": p_format(p)})
        for level in table.index:
            rows.append({"Characteristic": f"   {level}", "p": "",
                         **{k: pct(int((g[var] == level).sum()), int(g[var].notna().sum()))
                            for k, g in groups.items()}})
        missing = {k: int(g[var].isna().sum()) for k, g in groups.items()}
        if missing["Overall"]:
            rows.append({"Characteristic": "   Missing", "p": "",
                         **{k: str(v) for k, v in missing.items()}})
    return pd.DataFrame(rows, columns=["Characteristic", "Overall", "EBV+", "EBV−", "p"])


def render_png(table: pd.DataFrame, out: Path) -> None:
    """Plain typeset table: hairline rules, bold section rows, right-aligned numbers."""
    n = len(table)
    fig, ax = plt.subplots(figsize=(10, 0.32 * (n + 1) + 0.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 0.6, -0.8)
    xs = {"Characteristic": 0.0, "Overall": 0.62, "EBV+": 0.76, "EBV−": 0.90, "p": 1.0}
    for col, x in xs.items():
        ax.text(x, -0.3, col, ha="left" if col == "Characteristic" else "right",
                va="center", fontsize=12, weight="bold", color=INK)
    ax.axhline(0.15, color=INK, linewidth=1)
    for i, row in table.iterrows():
        section = not row["Characteristic"].startswith(" ") and row["Overall"] == ""
        for col, x in xs.items():
            ax.text(x, i + 0.6, row[col], ha="left" if col == "Characteristic" else "right",
                    va="center", fontsize=11, color=INK if col != "p" else SECONDARY,
                    weight="bold" if section and col == "Characteristic" else "normal")
    ax.axhline(n + 0.6 - 0.05, color=RULE, linewidth=1)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path,
                   default=PROJECT / "results/tcga2chile_binary_s0_46167280/splits.csv",
                   help="Any TCGA/Chile run's splits.csv: defines the analysed slides.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(PROJECT / "metadata/cohort_labels.csv")

    cohorts = {
        "tcga": (tcga_patients(labels, args.splits), {
            "Sex": ["Male", "Female"],
            "Molecular subtype": ["EBV", "MSI", "CIN", "GS", "POLE"],
            "MSI status": ["MSI-high", "MSS"],
            "Lauren type (from histological type)": ["Intestinal", "Diffuse",
                                                     "Adenocarcinoma NOS"],
            "Pathologic stage": ["I/II", "III/IV"],
        }, {"Molecular subtype", "MSI status"}),   # EBV status is defined by subtype
        "chile": (chile_patients(labels, args.splits), {
            "Sex": ["Male", "Female"],
            "Lauren type": ["Intestinal", "Diffuse", "Mixed", "Undetermined"],
            "Histologic grade": ["G1", "G2", "G3"],
            "Pathologic stage": ["I/II", "III/IV"],
        }, set()),
    }
    for name, (pts, orders, skip) in cohorts.items():
        table = build_table(pts, orders, skip)
        stem = args.out_dir / f"cohort_table_{name}"
        table.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")  # BOM: Excel-safe
        render_png(table, stem.with_suffix(".png"))
        print(f"\n== {name.upper()} ==\n{table.to_string(index=False)}")
    print(f"\nwrote cohort_table_{{tcga,chile}}.csv/.png to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
