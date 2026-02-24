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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import approaches as m
from profile import plot_approach_profile


def print_console(violations: list[m.ApproachAnalysis]) -> None:
    """Print violations as a formatted table."""
    print(f"\nAPPROACHES WITH +V GUIDANCE CLEARANCE < {m.TERPS_MIN_CLEARANCE_FT}'\n")

    if not violations:
        print("No violations found.")
        return

    # Column definitions: (header, width, right_justify, format_fn)
    cols = [
        ("Airport",  7, False, lambda a: a.airport),
        ("Approach", 28, False, lambda a: a.apch_name),
        ("+V Clr",   7, True, lambda a: str(a.worst_clearance_ft)),
        ("Proc Clr", 8, True, lambda a: str(a.lnav_clearance_ft)
         if a.lnav_clearance_ft is not None else ""),
        ("GPA",      5, True, lambda a: f"{a.gpa_deg:.1f}"),
    ]

    header = "  ".join(h.rjust(w) if rj else h.ljust(w)
                       for h, w, rj, _ in cols)
    sep = "  ".join("-" * w for _, w, _, _ in cols)
    print(header)
    print(sep)

    for a in violations:
        line = "  ".join(fmt(a).rjust(w) if rj else fmt(a).ljust(w)
                         for _, w, rj, fmt in cols)
        if (a.lnav_clearance_ft is not None
                and a.lnav_clearance_ft < m.TERPS_MIN_CLEARANCE_FT):
            line += "  ***"
        print(line)

    print(f"\nTotal: {len(violations)} approaches")


def generate_html(violations: list[m.ApproachAnalysis],
                  db: m.ApproachDatabase, output_dir: Path,
                  terrain) -> None:
    """Generate HTML report with sortable table and per-approach profile charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    all_legs = None
    n_cached = 0
    n_violations = len(violations)
    table_rows = []

    for i, analysis in enumerate(violations):
        apt = analysis.airport
        proc_id = analysis.proc_id
        apch_name = analysis.apch_name
        clearance = analysis.worst_clearance_ft
        gpa = analysis.gpa_deg

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
            if all_legs is None:
                all_legs = db.get_approach_legs()
            print(f"  [{i+1}/{n_violations}] {apt} {apch_name}...")
            apch_legs = all_legs[
                (all_legs["airport"] == apt) &
                (all_legs["proc_id"] == proc_id)]

            if not apch_legs.empty:
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                plot_approach_profile(apch_legs, terrain, ax, analysis)
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

        table_rows.append(
            f"<tr>"
            f"<td>{html.escape(apt)}</td>"
            f"<td>{html.escape(apt_name)}</td>"
            f"<td>{html.escape(state)}</td>"
            f"<td>{html.escape(apch_name)}</td>"
            f"{_clearance_td(clearance)}"
            f"{_clearance_td(analysis.lnav_clearance_ft)}"
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
    parser.add_argument("--srtm3", action="store_true",
                        help="use lower-res SRTM3 (90m) terrain "
                             "(default: SRTM1 30m)")
    args = parser.parse_args()

    airport = args.airport.upper() if args.airport else None

    db = m.ApproachDatabase()
    approaches = db.get_approaches(airport=airport)
    if approaches.empty:
        print("No approach geometries extracted.")
        sys.exit(1)

    terrain = db.get_terrain(srtm1=not args.srtm3)
    analyses = terrain.check_clearance(approaches, db)

    # Filter to violations
    violations = [a for a in analyses
                  if a.worst_clearance_ft is not None
                  and a.worst_clearance_ft < m.TERPS_MIN_CLEARANCE_FT]
    violations.sort(key=lambda a: a.worst_clearance_ft or 0)

    if args.html:
        generate_html(violations, db, Path("html_output"),
                      terrain=terrain)
    else:
        print_console(violations)


if __name__ == "__main__":
    main()
