#!/usr/bin/env python3
"""
Find all US RNAV (GPS) approaches where the +V advisory glidepath
violates the TERPS 250' minimum terrain clearance.

Usage:
    python approach_study.py              # all US approaches, console output
    python approach_study.py KSBS         # single airport, console output
    python approach_study.py --html       # all US approaches, HTML report
    python approach_study.py --html KSBS  # single airport, HTML report
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import approaches as m
from profile import plot_approach_profile


def add_lnav_clearance(violations: pd.DataFrame,
                       all_legs: pd.DataFrame,
                       db: m.ApproachDatabase) -> None:
    """Compute LNAV staircase clearance at each violation's worst +V point.

    Adds 'lnav_clearance_ft' column to violations in-place.
    """
    lnav_clearances = []
    for row in violations.itertuples():
        apch_legs = all_legs[
            (all_legs["airport"] == row.airport) &
            (all_legs["proc_id"] == row.proc_id)
        ]
        if apch_legs.empty:
            lnav_clearances.append(None)
            continue

        staircase = m.build_lnav_staircase(apch_legs, db=db)
        lnav_alt = staircase.altitude_at(row.worst_dist_nm)
        if lnav_alt is not None:
            lnav_clearances.append(int(round(lnav_alt - row.worst_terrain_ft)))
        else:
            lnav_clearances.append(None)

    violations["lnav_clearance_ft"] = lnav_clearances


def print_console(violations: pd.DataFrame) -> None:
    """Print violations as a formatted table."""
    print(f"\nAPPROACHES WITH +V GUIDANCE CLEARANCE < {m.TERPS_MIN_CLEARANCE_FT}'\n")

    if violations.empty:
        print("No violations found.")
        return

    # Column definitions: (header, width, right_justify, format_fn)
    cols = [
        ("Airport",  7, False, lambda r: r.airport),
        ("Approach", 28, False, lambda r: r.apch_name),
        ("+V Clr",   7, True, lambda r: str(r.worst_clearance_ft)),
        ("Proc Clr", 8, True, lambda r: str(r.lnav_clearance_ft)
         if r.lnav_clearance_ft is not None else ""),
        ("GPA",      5, True, lambda r: f"{r.gpa_deg:.1f}"),
    ]

    header = "  ".join(h.rjust(w) if rj else h.ljust(w)
                       for h, w, rj, _ in cols)
    sep = "  ".join("-" * w for _, w, _, _ in cols)
    print(header)
    print(sep)

    for row in violations.itertuples():
        line = "  ".join(fmt(row).rjust(w) if rj else fmt(row).ljust(w)
                         for _, w, rj, fmt in cols)
        if (row.lnav_clearance_ft is not None
                and row.lnav_clearance_ft < m.TERPS_MIN_CLEARANCE_FT):
            line += "  ***"
        print(line)

    print(f"\nTotal: {len(violations)} approaches")


def generate_html(violations: pd.DataFrame, db: m.ApproachDatabase,
                  output_dir: Path, all_legs: pd.DataFrame | None = None,
                  srtm1: bool = False) -> None:
    """Generate HTML report with sortable table and per-approach profile charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    chart_terrain = None
    n_cached = 0

    n_violations = len(violations)
    table_rows = []

    for i, row in enumerate(violations.itertuples()):
        apt = row.airport
        proc_id = row.proc_id
        apch_name = row.apch_name
        clearance = row.worst_clearance_ft
        gpa = row.gpa_deg

        # Airport metadata
        info = db.get_airport_info(apt)
        apt_name = info["name"]
        state = info["state"]

        # Generate chart (skip if already exists)
        chart_filename = f"{apt}_{proc_id}.png"
        chart_path = charts_dir / chart_filename

        if chart_path.exists() and chart_path.stat().st_size > 0:
            n_cached += 1
        else:
            print(f"  [{i+1}/{n_violations}] {apt} {apch_name}...")
            if chart_terrain is None:
                if all_legs is None:
                    print("Loading approach legs for chart generation...")
                    all_legs = db.get_approach_legs()
                res_label = "SRTM1 30m" if srtm1 else "SRTM3 90m"
                print(f"Initializing terrain for profile charts ({res_label})...")
                chart_terrain = db.get_terrain(srtm1=srtm1)

            assert all_legs is not None
            apt_legs = all_legs[all_legs["airport"] == apt]
            apch_legs = apt_legs[apt_legs["proc_id"] == proc_id]

            if not apch_legs.empty:
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                plot_approach_profile(apch_legs, chart_terrain, ax, db=db)
                plt.tight_layout()
                fig.savefig(chart_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

        if chart_path.exists():
            chart_link = (f'<a class="chart-link" '
                          f'href="charts/{html.escape(chart_filename)}" '
                          f'target="_blank">View</a>')
        else:
            chart_link = ""

        # Clearance styling helper
        def _clearance_td(val):
            if val is None:
                return "<td></td>"
            if val < 0:
                return f'<td class="clearance-negative">{val}</td>'
            if val < m.TERPS_MIN_CLEARANCE_FT:
                return f'<td class="clearance-warning">{val}</td>'
            return f"<td>{val}</td>"

        lnav_clearance = row.lnav_clearance_ft

        table_rows.append(
            f"<tr>"
            f"<td>{html.escape(apt)}</td>"
            f"<td>{html.escape(apt_name)}</td>"
            f"<td>{html.escape(state)}</td>"
            f"<td>{html.escape(apch_name)}</td>"
            f"{_clearance_td(clearance)}"
            f"{_clearance_td(lnav_clearance)}"
            f"<td>{gpa:.2f}</td>"
            f"<td>{chart_link}</td>"
            f"</tr>"
        )

    # Build HTML from template
    template_path = Path(__file__).parent / "report_template.html"
    template = template_path.read_text()

    summary = (f"<strong>{n_violations}</strong> approaches with +V advisory "
               f"glidepath clearance &lt; {m.TERPS_MIN_CLEARANCE_FT}&prime;")

    output_html = (template
                   .replace("{{SUMMARY}}", summary)
                   .replace("{{TABLE_ROWS}}", "\n".join(table_rows)))

    index_path = output_dir / "index.html"
    index_path.write_text(output_html)
    n_generated = n_violations - n_cached
    if n_cached:
        print(f"  {n_cached} charts cached, {n_generated} generated")
    print(f"\nHTML report: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find RNAV (GPS) approaches where +V advisory glidepath "
                    "violates TERPS 250' minimum terrain clearance.")
    parser.add_argument("airport", nargs="?", default=None,
                        help="ICAO airport code (e.g. KSBS); omit for all US")
    parser.add_argument("--html", action="store_true",
                        help="generate HTML report with sortable table and "
                             "profile charts")
    parser.add_argument("--srtm1", action="store_true",
                        help="use high-res SRTM1 (30m) terrain for profile "
                             "charts (default: SRTM3 90m)")
    args = parser.parse_args()

    airport = args.airport.upper() if args.airport else None

    db = m.ApproachDatabase()
    approaches = db.get_approaches(airport=airport)
    if approaches.empty:
        print("No approach geometries extracted.")
        sys.exit(1)

    terrain = db.get_terrain()
    violations = terrain.check_clearance(approaches)

    # Compute LNAV staircase clearance at each violation's worst +V point
    if not violations.empty:
        all_legs = db.get_approach_legs(airport=airport)
        add_lnav_clearance(violations, all_legs, db)
    else:
        all_legs = None

    if args.html:
        generate_html(violations, db, Path("html_output"),
                      all_legs=all_legs, srtm1=args.srtm1)
    else:
        print_console(violations)


if __name__ == "__main__":
    main()
